import os, sys, logging
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
#sys.setrecursionlimit(20000)
import time
import json
from glob import glob
import pandas as pd
from collections import defaultdict
from copy import deepcopy
import torch
import numpy as np
from tqdm import tqdm
import math
import shutil
import torch.distributed as dist
from multiprocessing import Pool
import multiprocessing as mp
import re
from functools import partial

import util
args = util.parser.parse_args()
if args.use_unsloth:
    try:
        from unsloth import FastLanguageModel
    except Exception as e:
        print(e)


from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Union


from sklearn.model_selection import train_test_split, GroupKFold, KFold, StratifiedKFold, StratifiedGroupKFold
from types import MethodType
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import RandomSampler, default_collate
from transformers import Trainer as HFTrainer, TrainerCallback, Seq2SeqTrainer as HFSeq2SeqTrainer, Seq2SeqTrainingArguments, TrainingArguments as HFTrainingArguments
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback, DataCollatorForSeq2Seq
from transformers import WhisperModel
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_utils import has_length
from transformers import AutoProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSpeechSeq2Seq, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.generation.utils import GenerationMixin
from transformers import ROPE_INIT_FUNCTIONS
from transformers import DebertaV2ForSequenceClassification
from transformers import LlamaForSequenceClassification, Qwen2ForSequenceClassification, Gemma2ForSequenceClassification, Qwen2Config
from transformers.models.t5 import T5ForSequenceClassification

from pt_util import set_seed
import pt_util as pu

from dataset import DatasetMixBase

logger = logging.getLogger(__name__)

# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
        shape: tuple[int, int],
        mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://huggingface.co/papers/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.detach().sum(-1).tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )
    print(111, input_lengths,  sequence_length, attention_mask is not None)

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)
    print(222, max_num_masked_span)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )
        print(999, num_masked_span, spec_aug_mask_idx)

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask

try:
    from transformers import CohereAsrForConditionalGeneration
    class CustomCohereAsrForConditionalGeneration(CohereAsrForConditionalGeneration):
        def _mask_input_features(
                self,
                input_features: torch.FloatTensor,
                attention_mask: Optional[torch.LongTensor] = None,
        ):
            """
            Masks extracted features along time axis and/or along feature axis according to
            [SpecAugment](https://huggingface.co/papers/1904.08779).
            """

            # `config.apply_spec_augment` can set masking to False
            # if not getattr(self.config, "apply_spec_augment", True):
            #    return input_features

            # generate indices & apply SpecAugment along time axis
            batch_size, hidden_size, sequence_length = input_features.size()

            if self.mask_time_prob > 0 and self.training:
                # generate indices & apply SpecAugment along time axis
                mask_time_indices = _compute_mask_indices(
                    (batch_size, sequence_length),
                    mask_prob=self.mask_time_prob,
                    mask_length=self.mask_time_length,
                    attention_mask=attention_mask,
                    min_masks=self.mask_time_min_masks,
                )
                mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
                mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
                input_features[mask_time_indices] = 0

            if self.mask_feature_prob > 0 and self.training:
                # generate indices & apply SpecAugment along feature axis
                mask_feature_indices = _compute_mask_indices(
                    (batch_size, hidden_size),
                    mask_prob=self.mask_feature_prob,
                    mask_length=self.mask_feature_length,
                    min_masks=self.mask_feature_min_masks,
                )
                mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
                input_features[mask_feature_indices] = 0

            return input_features

        def forward(
                self,
                input_features: torch.FloatTensor | None = None,
                attention_mask: torch.LongTensor | None = None,
                **kwargs,
        ):
            if self.training and (self.mask_feature_prob > 0 or self.mask_time_prob > 0):
                print(333, input_features.shape)
                input_features = self._mask_input_features(input_features, attention_mask=attention_mask)
            return super().forward(input_features, attention_mask, **kwargs)



        pass
except:
    pass

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


ppt_csrw = """"""
ppt_csrp = """Transcribe the following audio to International Phonetic Alphabet (IPA):"""


class DynamicBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        """
        dataset: The dataset with a 'lengths' attribute.
        max_tokens: Maximum token budget per batch (batch_size * max_len_in_batch <= max_tokens).
        shuffle: If True, shuffle before sorting (for randomness across epochs).
        """
        self.dataset = dataset
        self.cfg = self.dataset.cfg
        self.indices = list(range(len(dataset)))
        self.max_secs = self.cfg.max_sec*self.cfg.val_batch_size


    def __iter__(self):
        batch = []
        secs = 0
        for idx in self.indices:
            rec = self.dataset.data[idx]
            if (secs+rec.audio_duration_sec)>self.max_secs:
                yield batch
                batch, secs = [], 0
            batch.append(idx)
            secs += rec.audio_duration_sec
        if batch:
            yield batch

    def __len__(self):
        # Approximate length (number of batches); not exact due to dynamic sizing
        return (len(self.dataset) + self.max_tokens - 1) // self.max_tokens  # Rough estimate

class Sampler(torch.utils.data.Sampler):
    def __init__(self, cfg, data_type, ds):
        self.cfg = cfg
        self.data_type = data_type
        self.ds = ds
        self.inds = np.arange(len(ds))
        assert len(self.inds) == len(self.ds.data)
        title_num = defaultdict(int)
        for rec in self.ds.data:
            title_num[rec.title] += 1
        title_weight = {k:1/v for k, v in title_num.items()}
        self.weights = [title_weight[rec.title] for rec in self.ds.data]
        self.weights = np.array(self.weights)/np.sum(self.weights)

        assert abs(1-sum(self.weights))<1e-10

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for ind in self.gen_inds():
            yield ind

    def gen_inds(self):
        if self.data_type=='train':
            inds = np.random.choice(self.inds, self.__len__(), p=self.weights)
        else:
            raise NotImplementedError(self.data_type)
        return inds



def gen_ds(args, data_type, data, use_dl=False, **kwargs):
    drop_last, shuffle, num_workers, sampler, batch_size, collate_func = False, False, args.n_dl_worker, None, args.batch_size, None
    if data_type=='train':
        ds_cls = globals()[args.ds_cls]
        drop_last, shuffle = True, True
    elif data_type=='val':
        ds_cls = globals()[args.val_ds_cls]
        batch_size = args.val_batch_size
    else:
        ds_cls = globals()[args.test_ds_cls]
        batch_size = args.val_batch_size
    logger.info('ds cls:%s', ds_cls)
    ds = ds_cls(args, data_type, data, **kwargs)
    collate_func = ds.collate

    if args.use_sampler and data_type == 'train':
        sampler = Sampler(args, data_type, ds)
        shuffle = False
    if use_dl:
         ds = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=shuffle,
                                          drop_last=drop_last, collate_fn=collate_func, sampler=sampler)
    return ds


class DatasetMix(DatasetMixBase):
    def __init__(self, cfg, data_type, data, tokenizer=None, model_config=None, processor=None):
        super().__init__(cfg, data_type, data, tokenizer=tokenizer, model_config=model_config, processor=processor)
        self.ppt = globals()[cfg.ppt] if cfg.ppt is not None else ppt_csrw
        with util.timer('preprocess'):
            self.data = self.preprocess_data(data)
        if not self.cfg.is_eval and 'zipa' not in self.cfg.model_name:
            self.img_transform = self.create_img_transform()
            self.audio_transform = self.create_audio_transform()
        else:
            self.img_transform = None
            self.audio_transform = None

    def create_audio_transform(self):
        import audiomentations as A
        trans = []
        if self.data_type == 'train':
            trans.append(A.AddGaussianSNR(min_snr_db=self.cfg.am_gs_db, p=self.cfg.am_gs)) if self.cfg.am_gs > 0 else None
            trans.append(A.AddBackgroundNoise(sounds_path=["../data/preprocessed/noise_16k"], min_snr_db=self.cfg.am_bn_noise_db, max_snr_db=self.cfg.am_bn_noise_max_db, p=self.cfg.am_bn_noise)) if self.cfg.am_bn_noise>0 else None
        else:
            pass
        if len(trans) > 0:
            trans = A.Compose(trans)
        else:
            trans = None
        return trans

    def create_img_transform(self):
        import albumentations as A
        import cv2
        trans = []
        if self.data_type == 'train':
            pass
#            trans.append(A.XYMasking(p=self.cfg.alb_mxy)) if self.cfg.alb_mxy> 0 else None
#            trans.append(A.HorizontalFlip(p=self.cfg.alb_fliph)) if self.cfg.alb_fliph > 0 else None
#            trans.append(A.VerticalFlip(p=self.cfg.alb_flipv)) if self.cfg.alb_flipv > 0 else None
#            trans.append(A.HueSaturationValue(hue_shift_limit=self.cfg.alb_hue_shift, sat_shift_limit=self.cfg.alb_sat_shift,
#                                              val_shift_limit=self.cfg.alb_val_shift, p=self.cfg.alb_hsv)) if self.cfg.alb_hsv > 0 else None
#            trans.append(A.RandomBrightnessContrast(brightness_limit=self.cfg.alb_brightness_limit, contrast_limit=self.cfg.alb_contrast_limit,
#                                                    p=self.cfg.alb_rbc)) if self.cfg.alb_rbc > 0 else None
#
#            trans.append(A.ColorJitter(brightness=0.2 * self.cfg.alb_colorj_strength, contrast=0.2 * self.cfg.alb_colorj_strength, saturation=0.2 * self.cfg.alb_colorj_strength,
#                                       hue=0.5 * self.cfg.alb_colorj_strength, p=self.cfg.alb_colorj)) if self.cfg.alb_colorj > 0 else None
#            trans.append(A.Blur(p=self.cfg.alb_blur)) if self.cfg.alb_blur > 0 else None
#            trans.append(A.GaussianBlur(blur_limit=self.cfg.alb_gblur_limit, sigma_limit=self.cfg.alb_gblur_sigma, p=self.cfg.alb_gblur)) if self.cfg.alb_gblur > 0 else None
#            trans.append(A.MedianBlur(p=self.cfg.alb_mblur)) if self.cfg.alb_mblur > 0 else None
#            trans.append(A.ToGray(p=self.cfg.alb_gray)) if self.cfg.alb_gray > 0 else None
#            trans.append(A.CLAHE(p=self.cfg.alb_clahe)) if self.cfg.alb_clahe > 0 else None
#            trans.append(A.RandomGamma(p=self.cfg.alb_gamma)) if self.cfg.alb_gamma > 0 else None
#            trans.append(A.ImageCompression(quality_range=(75, 100), p=self.cfg.alb_ic)) if self.cfg.alb_ic > 0 else None
#            trans.append(A.Affine(scale=self.cfg.alb_affine_scale, rotate=self.cfg.alb_affine_rotate, shear=self.cfg.alb_affine_shear, translate_percent=self.cfg.alb_affine_trans,
#                                  p=self.cfg.alb_affine)) if self.cfg.alb_affine > 0 else None
        else:
            pass
        if len(trans) > 0:
            trans = A.Compose(trans)
            trans.set_random_seed(self.cfg.seed)
        else:
            trans = None
        return trans

    def preprocess_data(self, data):
        if self.data_type=='train' and 'wavelm' in self.cfg.model_name:
            data = data[~data.utterance_id.isin(['U_4196cd46d68a255f', 'U_8b8018a8a4169941', 'U_a78df694ce96d42d', 'U_b6cec1d8eaa22451', 'U_ce4a62974372080c', 'U_d3527c0d80eba26c', 'U_d8621dcc4b63fbf3', 'U_e996d944bb561329', 'U_eb7dcecc7a6d07e5', 'U_f14080cc550fcdf1'])]
            data['phonetic_text'] = data.phonetic_text.apply(lambda x: re.sub(' +', ' ', x))
            if self.cfg.dataset.startswith('ipap'):
                table = str.maketrans("", "", 'ɵ̴øaɜɕœɯʲỹɤɦɲʈʏʰɥʂ˞')
                data['phonetic_text'] = data.phonetic_text.apply(lambda x: x.translate(table))
                num = len(data)
                data = data[data.phonetic_text.apply(lambda x: len(x)>0)]
                logger.info('%s removed empty:%s', self.cfg.dataset, num-len(data))
        if self.cfg.aug_clone>0:
            data['text_id'] = data.orthographic_text.apply(util.get_text_id_hash)
            aaa = pd.read_csv('../data/clone_data.csv')
            aaa = aaa[aaa.audio_duration_sec < (self.cfg.max_sec or 1000000)]
            aaa = {text_id: g.to_records(index=False) for text_id, g in aaa.groupby('text_id')}
            self.clone_data = aaa
            logger.info('clone data:%s', len(self.clone_data))
        if self.data_type!='train' and 'audio_duration_sec' in data.columns:
                data = data.sort_values(['audio_duration_sec'], ascending=False)
        data = super().preprocess_data(data)
        if self.cfg.use_gen_audio and self.data_type=='train':
            gen_audios = pd.read_csv(f"../data/gen_audios.csv")
            gen_audios = gen_audios[gen_audios.audio_duration_sec < (self.cfg.max_sec or 1000000)]
            gen_audios = gen_audios.rename(columns={"text": "orthographic_text", 'text_id': 'utterance_id'})
            gen_audios['src'] = 'gen_audio'
            logger.info('num of gen_audios:%s', len(gen_audios))
            data = list(data) + list(gen_audios.to_records(index=False))
        if self.data_type=='train' and self.cfg.aug_cat>0:
            self.audio_lens = defaultdict(list)
            for rec in data:
                self.audio_lens[rec.audio_duration_sec//5].append(rec)
            self.audio_lens_keys = sorted(self.audio_lens.keys())
        return data

    def aug_audio(self, audio, sr):
        if self.audio_transform is not None:
            audio = self.audio_transform(samples=audio, sample_rate=sr)
        return audio, sr

    def getitem(self, index, rec=None, **kwargs):
        item = self._getitem(index, rec=rec, **kwargs)
        if self.data_type=='train' and (len(item['audio'])/self.cfg.sr)<(self.cfg.max_sec//2) and self.cfg.aug_cat>0 and np.random.rand()<self.cfg.aug_cat:
            l = self.cfg.max_sec - len(item['audio'])/self.cfg.sr
            l = l//5
            ks = []
            for k in self.audio_lens_keys:
                if k<=l:
                    ks.append(k)
                else:
                    break
            if len(ks)>0:
                k = np.random.choice(ks)
                cat_rec = np.random.choice(self.audio_lens[k])
                cat_item = self._getitem(None, rec=cat_rec)
                item = self.cat_item(item, cat_item)

        if self.data_type=='train' and self.cfg.mixup>0 and np.random.rand()<self.cfg.mixup:
            mixup_item = self.sample_item(index)
            item['mixup_item'] = mixup_item

        return item

    def get_audio(self, rec):
        if rec.src.startswith('libri'):
            offset = rec.start
            duration = rec.duration
        else:
            offset = 0
            duration = None
        try:
            if self.data_type=='train' and self.cfg.aug_clone>0 and rec.text_id in self.clone_data and np.random.rand()<self.cfg.aug_clone:
                audio_path = np.random.choice(self.clone_data[rec.text_id]).audio_path

            else:
                audio_path = rec.audio_path
            audio, sr = util.load_audio(audio_path, sr=self.cfg.sr, offset=offset, duration=duration)
        except Exception as e:
            logger.error("audio:%s, error:%s", rec.audio_path, e)
            raise e
        if rec.src.startswith('semi'):
            step = sr*self.cfg.max_sec
            audio = audio[rec.split_id*step:(rec.split_id+1)*step]
        else:
            if duration is None:
                audio = audio[:sr * self.cfg.max_sec]
        return audio, sr

    def _getitem(self, index, rec=None):
        if rec is None:
            rec = self.data[index]
        audio, sr = self.get_audio(rec)
        if self.data_type=='train':
            audio, sr = self.aug_audio(audio, sr)
        item = dict(audio=audio)
        if self.data_type!='test':
            item['labels'] = self.tokenizer.encode(rec.orthographic_text)
            item['text'] = rec.orthographic_text
        return item

    def collate(self, batch):
        new_batch = dict()
        for k in ['utterance_id', 'text', 'duration']:
            if k in batch[0]:
                new_batch[k] = [item.pop(k) for item in batch]

        input_lens = [item['seq_len'] for item in batch]
        max_len = max(input_lens)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        for i, item in enumerate(batch):
            if self.tokenizer.padding_side == 'left':
                if 'input_ids' in item:
                    item['input_ids'] = np.pad(item['input_ids'], ((max_len - item['seq_len'], 0)), "constant", constant_values=pad_token_id)
                if 'attention_mask' in item:
                    item['attention_mask'] = np.pad(item['attention_mask'], ((max_len - item['seq_len'], 0)), "constant", constant_values=0)
                if 'labels' in item and not self.cfg.is_classify and not self.cfg.use_soft1:
                    item['labels'] = np.pad(item['labels'], ((max_len - item['seq_len'], 0)), "constant", constant_values=-100)
            else:
                if 'input_ids' in item:
                    item['input_ids'] = np.pad(item['input_ids'], ((0, max_len - item['seq_len'])), "constant", constant_values=pad_token_id)
                if 'attention_mask' in item:
                    item['attention_mask'] = np.pad(item['attention_mask'], ((0, max_len - item['seq_len'])), "constant", constant_values=0)
                if 'labels' in item and not self.cfg.is_classify and not self.cfg.use_soft1:
                    item['labels'] = np.pad(item['labels'], ((0, max_len - item['seq_len'])), "constant", constant_values=-100)
        batch = default_collate(batch)
        batch.update(new_batch)
        return batch


class Dataset(DatasetMix, torch.utils.data.Dataset):
    pass


class TransMix():
    def getitem(self, index, rec=None, **kwargs):
        item = self._getitem(index, rec=rec, **kwargs)
        return item

    def _getitem(self, index, rec=None):
        if rec is None:
            rec = self.data[index]
        item = dict(utterance_id=rec.utterance_id, duration=rec.audio_duration_sec)
        text = rec.orthographic_text
        msg = [{"role": "user", "content": f'''Please translate below sentence to International Phonetic Alphabet (IPA):\n{text}'''}]
        input_ids = np.array(self.tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=False))
        l = len(input_ids)
        if self.data_type!="test" and not self.cfg.is_eval:
            ipa = rec.phonetic_text
            msg.append({"role": "assistant", "content": ipa})
            input_ids = np.array(self.tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=False, enable_thinking=False))
            labels = deepcopy(input_ids)
            labels[:l] = -100
            item['labels'] = labels
        item['input_ids'] = input_ids
        item['seq_len'] = len(item['input_ids'])
        if hasattr(rec, 'split_id'):
            item['split_id'] = rec.split_id
        return item

class TransDataset(TransMix, Dataset):
    pass


class QWASRMix():
    def __init__(self, cfg, data_type, data, tokenizer=None, model_config=None, processor=None):
        super().__init__(cfg, data_type, data, tokenizer=tokenizer, model_config=model_config, processor=processor)
        prefix_msgs = [
            {"role": "system", "content": self.ppt},
            {"role": "user", "content": [{"type": "audio", "audio": None}]},
        ]
        if 'wavelm' not in self.cfg.model_name and 'cohere' not in self.cfg.model_name and 'trans' not in self.cfg.model_name:
            self.prefix_text = self.processor.apply_chat_template(
                [prefix_msgs], add_generation_prompt=True, tokenize=False
            )[0]

        if 'granite' in self.cfg.model_name:
            instruction = "Please transcribe the following audio to text<|audio|>"
            chat = [dict(role="user", content=instruction)]
            text = tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            )
            self.granite_prompt = text

    def _getitem(self, index, rec=None, audio=None, sr=None):
        if rec is None:
            rec = self.data[index]
        if audio is None:
            audio, sr = self.get_audio(rec)
        if self.data_type=='train':
            audio, sr = self.aug_audio(audio, sr)
        item = dict(audio=audio, utterance_id=rec.utterance_id, duration=rec.audio_duration_sec)
        if self.data_type != 'test':
            item['text'] = rec['orthographic_text'] if self.cfg.model_name.startswith('CSRW') else rec['phonetic_text']
        return item

    def cat_item(self, item, cat_item):
        item['text'] = item['text'] + " " + cat_item['text']
        item['audio'] = np.concatenate([item['audio'], cat_item['audio']])
        return item

    def get_iter_items(self, index):
        rec = self.data[index]
        try:
            audio, sr = util.load_audio(rec.audio_path, sr=self.cfg.sr)
        except Exception as e:
            logger.error("audio:%s, error:%s", rec.audio_path, e)
            raise e

        for i, s in enumerate(range(0, len(audio), sr*self.cfg.max_sec)):
            split_audio = audio[s:s+sr * self.cfg.max_sec]
            item = self.getitem(index, rec=rec, audio=split_audio, sr=sr)
            item['split_id'] = i
            item['duration'] = len(split_audio)/sr
            if self.cfg.min_sec is None or i==0:
                yield item
            else:
                if item['duration'] >= self.cfg.min_sec:
                    yield item
    def collate_granite(self, batch):
        utterance_ids = [f["utterance_id"] for f in batch]
        durations = [f["duration"] for f in batch]
        split_ids = [f.get("split_id", None) for f in batch]

        prompts = [self.granite_prompt for item in batch]
        audios = [item["audio"] for item in batch]
        processed = self.processor(prompts, audios, return_tensors="pt", padding=True, padding_side="left")
        input_ids = processed.input_ids
        attention_mask = processed.attention_mask
        labels = None
        if self.data_type!='test':
            targets = [item["text"] + self.processor.tokenizer.eos_token for item in batch]
            targets = self.processor.tokenizer(targets, return_tensors="pt", padding=True, padding_side="right")
            # combine prompt+targets
            input_ids = torch.cat([input_ids, targets.input_ids], dim=1)
            attention_mask = torch.cat([attention_mask, targets.attention_mask], dim=1)
            labels = targets.input_ids.clone()
            # Set non-target tokens to -100 for loss calculation
            labels[~(targets.attention_mask.bool())] = -100
            labels = torch.cat([torch.full_like(processed.input_ids, -100), labels], dim=1)
        batch = dict(input_ids=input_ids, attention_mask=attention_mask)
        if labels is not None:
            batch['labels'] = labels
        if self.data_type != 'train':
            batch['utterance_id'] = utterance_ids
            batch['duration'] = durations
            if split_ids[0] is not None:
                batch['split_id'] = split_ids
        return batch


    def collate_wavelm(self, batch):
        utterance_ids = [f["utterance_id"] for f in batch]
        durations = [f["duration"] for f in batch]
        split_ids = [f.get("split_id", None) for f in batch]

        audios = [item["audio"] for item in batch]
        input_values = [{"input_values": self.processor(audio, sampling_rate=self.cfg.sr).input_values[0]} for audio in audios]
        batch_feature = self.processor.feature_extractor.pad(
            input_values,
            padding=True,
            return_tensors="pt",
        )
        if self.data_type!='test' and not self.cfg.is_eval:
            input_ids = [{"input_ids": self.processor.tokenizer(item['text']).input_ids} for item in batch]
            input_ids_batch = self.processor.tokenizer.pad(input_ids, padding=True, return_tensors="pt", )
            batch_feature['labels'] = input_ids_batch['input_ids'].masked_fill(input_ids_batch.attention_mask.ne(1), -100)
        batch = batch_feature

        if self.data_type != 'train':
            batch['utterance_id'] = utterance_ids
            batch['duration'] = durations
            if split_ids[0] is not None:
                batch['split_id'] = split_ids
        return batch

    def collate(self, batch):
        if 'granite' in self.cfg.model_name:
            return self.collate_granite(batch)
        elif 'wavelm' in self.cfg.model_name:
            return self.collate_wavelm(batch)
        elif 'cohere' in self.cfg.model_name:
            return self.collate_cohere(batch)
        num = len(batch)
        mixup_batch = []
        mixup_weights = []
        if self.data_type=='train' and any(['mixup_item' in item for item in batch]):
            for item in batch:
                if 'mixup_item' in item:
                    mixup_batch.append(item['mixup_item'])
                    mixup_weights.append(np.random.beta(1, 1))
                else:
                    mixup_batch.append(item)
                    mixup_weights.append(1)
            assert len(mixup_batch) == num
        batch = batch + mixup_batch
        batch = self._collate(batch)
        if len(mixup_batch)>0:
            mixup_batch = {k: batch[k][-num:] for k in batch}
            batch = {k: batch[k][:num] for k in batch}
            for i, w in enumerate(mixup_weights):
                if w!=1:
                    batch['input_features'][i] = batch['input_features'][i] *w + mixup_batch['input_features'][i] * (1-w)
            batch['mixup_input_features'] = deepcopy(batch['input_features'])
            batch['mixup_input_ids'] = mixup_batch['input_ids']
            batch['mixup_labels'] = mixup_batch['labels']
            batch['mixup_attention_mask'] = mixup_batch['attention_mask']
            batch['mixup_feature_attention_mask'] = mixup_batch['feature_attention_mask']
            batch['mixup_w'] = torch.from_numpy(np.array(mixup_weights))
        return batch

    def collate_cohere(self, batch):
        utterance_ids = [f["utterance_id"] for f in batch]
        durations = [f["duration"] for f in batch]
        split_ids = [f.get("split_id", None) for f in batch]
        audios = [f["audio"] for f in batch]
        inputs = self.processor(audios, sampling_rate=self.cfg.sr, return_tensors="pt", language="en", punctuation=False)
        if self.data_type!='test' and not self.cfg.is_eval:
            texts = [f['text'] for f in batch]
            inputs = self.processor(audios, text=texts, sampling_rate=self.cfg.sr, padding=True, return_tensors="pt", language="en", punctuation=False)
            decoder_input_ids = inputs['decoder_input_ids']
            print(111, inputs['attention_mask'].sum(axis=-1), inputs['input_features'].shape)
            prefix_len = decoder_input_ids.shape[1]
            labels = inputs['labels']
            decoder_input_ids = torch.cat([decoder_input_ids, labels], axis=-1)
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
            decoder_attention_mask[decoder_attention_mask==self.tokenizer.pad_token_id] = 0
            inputs['decoder_input_ids'] = decoder_input_ids
            inputs['decoder_attention_mask'] = decoder_attention_mask
            labels = decoder_input_ids.clone()
            labels[:, :prefix_len] = -100
            inputs['labels'] = labels
        if self.data_type != 'train':
            inputs['utterance_id'] = utterance_ids
            inputs['duration'] = durations
            if split_ids[0] is not None:
                inputs['split_id'] = split_ids
        return inputs

    def _collate(self, features):
        audios = [f["audio"] for f in features]
        prefix_texts = [self.prefix_text + 'language English<asr_text>' for f in features]
        utterance_ids = [f["utterance_id"] for f in features]
        durations = [f["duration"] for f in features]
        split_ids = [f.get("split_id", None) for f in features]

        padding_side = 'left' if self.cfg.is_eval else 'right'

        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
            padding_side=padding_side,
        )
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()

        if not self.cfg.is_eval:
            targets = [f["text"] for f in features]
            eos = self.processor.tokenizer.eos_token or ""
            full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

            full_inputs = self.processor(
                text=full_texts,
                audio=audios,
                return_tensors="pt",
                padding=True,
                truncation=False,
                padding_side=padding_side,
            )

            labels = full_inputs["input_ids"].clone()
            for i, pl in enumerate(prefix_lens):
                labels[i, :pl] = -100

            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100

            full_inputs["labels"] = labels
        else:
            full_inputs = prefix_inputs
        if self.data_type!='train':
            full_inputs['utterance_id'] = utterance_ids
            full_inputs['duration'] = durations
            if split_ids[0] is not None:
                full_inputs['split_id'] = split_ids
        if self.cfg.fix_continuous:
            full_inputs['attention_mask'] = full_inputs['attention_mask'].contiguous()
        return full_inputs


class QWASRDataset(QWASRMix, Dataset):
    pass



class IterMix(DatasetMix):
    def __len__(self):
        return len(self.data)

    def get_distribute_data(self, data, world_size=None, rank=None):
        rank = dist.get_rank() if rank is None else rank
        world_size = dist.get_world_size() if world_size is None else world_size
        per_rank = int(math.ceil(len(data) / world_size))
        return data[rank * per_rank:(rank + 1) * per_rank]

    def get_iter_items(self, index):
        rec = self.data[index]
        yield self.getitem(index, rec=rec)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        data = self.data
        if worker_info is not None:
            self.data = self.get_distribute_data(data, worker_info.num_workers, worker_info.id)
        for index in range(len(self.data)):
            items = self.get_iter_items(index)
            for item in items:
                if item is not None:
                    yield item

class IterDataset(IterMix, torch.utils.data.IterableDataset):
    pass


class AudioMix():
    def __init__(self, cfg, data_type, data, tokenizer=None, model_config=None, processor=None):
        super().__init__(cfg, data_type, data, tokenizer=tokenizer, model_config=model_config, processor=processor)
        from lhotse.features.kaldi.extractors import Fbank
        self.fbank = Fbank()

    def get_fbank(self, audio):
        features = self.fbank.extract_batch(
            audio, sampling_rate=16000
        )
        feature_lens = torch.tensor([len(feature) for feature in features])
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        return features, feature_lens

    def get_iter_items(self, index):
        rec = self.data[index]
        try:
            audio, sr = util.load_audio(rec.audio_path, sr=self.cfg.sr)
        except Exception as e:
            logger.error("audio:%s, error:%s", rec.audio_path, e)
            raise e

        for i, s in enumerate(range(0, len(audio), sr*self.cfg.max_sec)):
            split_audio = audio[s:s+sr * self.cfg.max_sec]
            item = dict(utterance_id=rec.utterance_id, audio=split_audio, sr=sr)
            item['split_id'] = i
            item['duration'] = len(split_audio)/sr
            yield item

    def collate(self, batch):
        new_batch = dict()
        for k in ['utterance_id', 'split_id', 'duration']:
            if k in batch[0]:
                new_batch[k] = [item.pop(k) for item in batch]
        features, feature_lens = self.get_fbank([torch.from_numpy(item['audio']) for item in batch])
        new_batch['feature'] = features
        new_batch['feature_lens'] = feature_lens
        return new_batch


class QWASRIterDataset(QWASRMix, IterDataset):
    pass


class AudioIterDataset(AudioMix, IterDataset):
    pass


class DynamicMix():
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        data = self.data
        if worker_info is not None:
            self.data = self.get_distribute_data(data, worker_info.num_workers, worker_info.id)
        batch, max_sec_in_batch = [], 0
        for index in range(len(self.data)):
            items = self.get_iter_items(index)
            for item in items:
                if item is not None:
                    new_max_sec = max(max_sec_in_batch, item['duration'])
                    #if batch and (len(batch)+1)*np.power(self.cfg.max_sec/new_max_sec, self.cfg.dynamic_power)*new_max_sec>=(self.cfg.max_sec*self.cfg.val_batch_size):
                    if batch and ((max_sec_in_batch>=30 and len(batch)>=96) or (max_sec_in_batch<30 and len(batch)>=self.cfg.val_batch_size)):
                        yield batch
                        batch, max_sec_in_batch = [], 0
                    batch.append(item)
                    max_sec_in_batch = new_max_sec
        if batch:
            yield batch


class QWASRDynamicDataset(DynamicMix, QWASRIterDataset):
    pass


class ModelEmaV2(torch.nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device=None, update_after_step: int = 0):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = model
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model, step=None):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

from torch.optim import Muon, AdamW, Optimizer

class MuonHybrid(Optimizer):
    def __init__(self, model, decay_parameters, lr, weight_decay, **kwargs):
        # We don't use the standard defaults dict because we are managing two optimizers
        self.muon_lr = lr
        self.adam_lr = lr*0.5
        exclude_from_muon = ["emb", "head"]

        muon_params = []
        adamw_decay_params = []
        adamw_no_decay_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # 1. Check if it's a 1D vector (Bias, LayerNorms)
            if p.ndim != 2:
                if name not in decay_parameters:
                    adamw_no_decay_params.append(p)
                else:
                    adamw_decay_params.append(p)

            # 2. Check if it's the Embedding or the LM Head
            elif any(ex in name.lower() for ex in exclude_from_muon):
                print(222, name)
                if name not in decay_parameters:
                    adamw_no_decay_params.append(p)
                else:
                    adamw_decay_params.append(p)
            elif name not in decay_parameters:
                adamw_no_decay_params.append(p)
            # 3. Everything else is a 2D Transformer Hidden Layer weight matrix
            else:
                print('muon', name)
                muon_params.append(p)
        adam_grouped_parameters = [
            {"params": adamw_decay_params, "lr": self.adam_lr, "weight_decay": weight_decay, },  # AdamW Standard
            {"params": adamw_no_decay_params, "lr": self.adam_lr, "weight_decay": 0.0},  # AdamW No Decay
        ]
        muon_grouped_parameters = [
            {"params": muon_params, "lr": self.muon_lr, "weight_decay": weight_decay, },  # AdamW Standard
        ]

        self.muon_opt = Muon(muon_grouped_parameters, lr=self.muon_lr, weight_decay=weight_decay, **kwargs)
        self.adamw_opt = AdamW(adam_grouped_parameters, lr=self.adam_lr, weight_decay=weight_decay)

        # Expose param_groups for Trainer/Scheduler compatibility
        self.param_groups = self.muon_opt.param_groups + self.adamw_opt.param_groups

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.muon_opt.step()
        self.adamw_opt.step()
        return loss

    def zero_grad(self, set_to_none=True):
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        self.adamw_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            'muon': self.muon_opt.state_dict(),
            'adamw': self.adamw_opt.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.muon_opt.load_state_dict(state_dict['muon'])
        self.adamw_opt.load_state_dict(state_dict['adamw'])
class TrainerMix():
    def __init__( self, model=None, args=None, **kwargs ):
        self.custom_start_time = time.time()
        self.curr_train_step = 0
        super().__init__(model=model, args=args, **kwargs)
        self.ema = None

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        logger.info('model saved to %s', output_dir)
        if self.ema is not None:
            self.ema.module.save_pretrained(f"{output_dir}")
            logger.info('ema model saved to %s',f"{output_dir}")
        model = self.accelerator.unwrap_model(self.model_wrapped)
        if self.args.save_lm_head:
            if hasattr(model, 'lm_head'):
                torch.save(model.lm_head.state_dict(), f"{output_dir}/lm_head.pt")
            else:
                torch.save(model.base_model.model.lm_head.state_dict(), f"{output_dir}/lm_head.pt")

    def log(self, logs, start_time=None):
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if 'elapsed' not in logs:
            logs['elapsed'] = time.time()-self.custom_start_time
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.args.use_muon:
            logger.info('use muon')
            return self.create_muon_optimizer()
        else:
            return super().create_optimizer()
        return self.optimizer

    def create_muon_optimizer(self):
        from torch.optim import Muon
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            # Parameters to exclude from weight decay (Standard Transformers logic)
            decay_parameters = self.get_decay_parameter_names(opt_model)
            self.optimizer = MuonHybrid(opt_model, decay_parameters, lr=self.args.learning_rate, weight_decay=self.args.weight_decay, momentum=0.95)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None
        if self.args.use_sampler>0:
            return Sampler(self.args, 'train', train_dataset)
        else:
            return RandomSampler(self.train_dataset)

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.args.ema>0:
            if self.ema is None:
                model = model.module if hasattr(model, "module") else model
                self.ema = ModelEmaV2(model=deepcopy(model), decay=self.args.ema, device='cpu')

            if self.accelerator.sync_gradients:
                model = model.module if hasattr(model, "module") else model
                self.ema.update(model)

        return loss


class Trainer(TrainerMix, HFTrainer):
    def __init__( self, model=None, args=None, **kwargs ):
        super().__init__(model=model, args=args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)


@dataclass
class TrainingArguments(HFTrainingArguments):
    rdrop: float = field(default=0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    semi_ratio: float = field(default=0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    temp: float = field(default=1, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    is_classify: bool = field(default=False, metadata={"help": "Whether to run training."})
    is_rm: bool = field(default=False, metadata={"help": "Whether to run training."})
    is_rmp: bool = field(default=False, metadata={"help": "Whether to run training."})
    save_lm_head: bool = field(default=False, metadata={"help": "Whether to run training."})
    model_name: str = field(default=False, metadata={"help": "Whether to run training."})
    use_soft1: bool = field(default=False, metadata={"help": "Whether to run training."})
    ema: float = field(default=0)

    use_adam_mini: bool = field(default=False, metadata={"help": "Whether to run training."})
    use_muon: bool = field(default=False, metadata={"help": "Whether to run training."})
    use_badam: bool = field(default=False, metadata={"help": "Whether to run training."})
    use_sampler: bool = field(default=False, metadata={"help": "Whether to run training."})
    switch_block_every: int = field(default=32, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."})
    hard_ratio: float = field( default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})


def load_unsloth_model(args, model_id):
    seed = args.unsloth_seed or args.seed
    kwargs = dict()
    logger.info('modelid %s', model_id)
    if args.is_eval and args.is_classify:
        model, tokenizer = FastLanguageModel.from_pretrained(args.backbone, dtype=getattr(torch, args.torch_dtype), use_cache=False, max_seq_length=args.max_seq_len+8,
                                                             load_in_4bit=args.use_4bit, full_finetuning=args.use_full, load_in_8bit=args.use_8bit,
                                                             use_gradient_checkpointing='unsloth' if args.gradient_checkpointing else False, **kwargs)
        FastLanguageModel.for_inference(model)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(model_id, dtype=getattr(torch, args.torch_dtype), use_cache=False, max_seq_length=args.max_seq_len+8+args.max_gen_len,
                                                             load_in_4bit=args.use_4bit, full_finetuning=args.use_full, load_in_8bit=args.use_8bit,
                                                             use_gradient_checkpointing='unsloth' if args.gradient_checkpointing else False, **kwargs)
    if args.use_lora:
        lora_init = args.lora_init or True
        target_modules = find_all_linear_names(args, model)
        model = FastLanguageModel.get_peft_model(model, r=args.lora_rank, lora_alpha=args.lora_alpha,
                                             lora_dropout=args.lora_dropout, bias="none",
                                             random_state=seed,
                                             use_gradient_checkpointing='unsloth' if args.gradient_checkpointing else False,
                                             target_modules=target_modules if args.lora_modules is None else args.lora_modules,
                                             use_dora=args.use_dora, init_lora_weights=lora_init)
    if args.is_classify:
        config = AutoConfig.from_pretrained(model_id)
        if config.architectures is not None and 'ForSequenceClassification' in config.architectures[0]:
            tmp_model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True, torch_dtype=getattr(torch, args.torch_dtype),
                                                                           num_labels=args.n_label)
            lm_head = tmp_model.score
            del tmp_model
            logger.info("create lm head from score")
        else:
            logger.info('hidden size:%s', config.hidden_size)
            lm_head = nn.Linear(model.config.hidden_size, args.n_label, bias=False)
            lm_head.to(getattr(torch, args.torch_dtype))
        logger.info('unsloth classify')
        if os.path.exists(f"{model_id}/lm_head.pt"):
            logger.info("restore score from lm head")
            state_dict = torch.load(f"{model_id}/lm_head.pt", weights_only=True)
            lm_head.load_state_dict(state_dict)
        if args.use_lora:
            model.base_model.model.lm_head = lm_head.to(model.base_model.model.lm_head.weight.device)
            logger.info('lm head dtype:%s', model.base_model.model.lm_head.weight.dtype)
        else:
            model.lm_head = lm_head.to(model.lm_head.weight.device)
            logger.info('lm head dtype:%s', model.lm_head.weight.dtype)
    return model, tokenizer

def find_all_linear_names(args, model):
    import bitsandbytes as bnb
    cls = bnb.nn.Linear4bit if args.use_4bit else (bnb.nn.Linear8bitLt if args.use_8bit else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    logger.info('find linear:%s', lora_module_names)


    return list(lora_module_names)



def load_model(args, model_id):
    model_id = util.get_modelid(model_id)
    if args.use_unsloth:
        model, tokenizer = load_unsloth_model(args, model_id)

    elif args.use_lora:
        lora_init = args.lora_init or True
        from peft import LoraConfig, get_peft_model
        import bitsandbytes as bnb
        if args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.use_4bit,
                load_in_8bit=args.use_8bit,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=getattr(torch, args.torch_dtype),
                bnb_4bit_use_double_quant=args.use_double_quant,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None
        kwargs = dict()
        if quantization_config is not None:
            kwargs['quantization_config'] = quantization_config
        if args.is_classify:
            model = AutoModelForSequenceClassification.from_pretrained(model_id, device_map={"": 0}, trust_remote_code=True,
                                                                       torch_dtype=getattr(torch, args.torch_dtype), num_labels=args.n_label, **kwargs)
            task_type = 'SEQ_CLS'
            util.restore_lm_head(args, model_id, model)

        else:
            if args.model_name.startswith('CSR'):
                if 'granite' in args.model_name:
                    model_cls = AutoModelForSpeechSeq2Seq
                    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                    tokenizer = processor.tokenizer
                elif 'wavelm' in args.model_name:
                    from transformers import WavLMForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
                    from score_func import VALID_IPA_CHARS
                    model_cls = WavLMForCTC
                    phonemes = sorted(VALID_IPA_CHARS)
                    logger.info('phonemes:%s', phonemes)
                    vocab = ["<pad>", "<s>", "</s>", "<unk>"] + phonemes
                    vocab_dict = {token: idx for idx, token in enumerate(vocab)}

                    with open("/tmp/wavelm_vocab.json", "w", encoding="utf-8") as f:
                        json.dump(vocab_dict, f, ensure_ascii=False)
                    tokenizer = Wav2Vec2CTCTokenizer(
                        vocab_file= '/tmp/wavelm_vocab.json',
                        unk_token="<unk>",
                        pad_token="<pad>",
                        word_delimiter_token=" ",
                        do_lower_case=False,
                    )

                    feature_extractor = Wav2Vec2FeatureExtractor(
                        sampling_rate=16000,
                        return_attention_mask=True,
                    )

                    processor = Wav2Vec2Processor(
                        feature_extractor=feature_extractor,
                        tokenizer=tokenizer,
                    )
                else:
                    model_cls = AutoModelForSpeechSeq2Seq
                    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                    tokenizer = processor.tokenizer

            else:
                model_cls = AutoModelForCausalLM
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = model_cls.from_pretrained(model_id, device_map={"": 0}, trust_remote_code=True,
                                                         torch_dtype=getattr(torch, args.torch_dtype), **kwargs)
            task_type = 'CAUSAL_LM'
        logger.info('model type:%s', type(model))
        if args.use_4bit:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        if 'KF' in model_id and hasattr(model, 'peft_config'):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_id, is_trainable=True)
        else:
            target_modules = find_all_linear_names(args, model)
            #layers_to_transform = [i for i in range(model.config.num_hidden_layers) if i >= args.lora_start_layer]
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules if args.lora_modules is None else args.lora_modules,
                #layers_to_transform=layers_to_transform,
                lora_dropout=args.lora_dropout,
                use_dora=args.use_dora,
                bias="none",
                task_type=task_type,
                #modules_to_save=args.modules_to_save,
                init_lora_weights=lora_init,
            )
            model.enable_input_require_grads()

            model = get_peft_model(model, lora_config)

    else:
        kwargs = dict()
        if args.is_classify:
            model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True,
                                        torch_dtype=getattr(torch, args.torch_dtype), num_labels=args.n_label)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        elif args.model_name.startswith('CSR'):
            if "qw3asr" in args.model_name:
                from qwen_asr import Qwen3ASRModel
                if args.audio_att_dropout>0 or args.text_att_dropout>0:
                    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                    config.thinker_config.audio_config.attention_dropout = args.audio_att_dropout
                    config.thinker_config.text_config.attention_dropout = args.text_att_dropout
                    asr_wrapper = Qwen3ASRModel.from_pretrained(model_id, config=config, torch_dtype=getattr(torch, args.torch_dtype), device_map=None)
                else:
                    asr_wrapper = Qwen3ASRModel.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), device_map=None)
                model = asr_wrapper.model
                if args.frozen_audio:
                    for param in model.thinker.audio_tower.parameters():
                        param.requires_grad = False
                if args.frozen_llm:
                    for param in model.thinker.model.parameters():
                        param.requires_grad = False

                model.generation_config = GenerationConfig.from_model_config(model.config)
                util.patch_outer_forward(model)
                processor = asr_wrapper.processor
                tokenizer = processor.tokenizer
            elif 'trans' in args.model_name:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), device_map=None)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                processor = tokenizer
            elif 'cohere' in args.model_name:
                from transformers import AutoProcessor, CohereAsrForConditionalGeneration
                processor = AutoProcessor.from_pretrained(model_id)
                model = CustomCohereAsrForConditionalGeneration.from_pretrained(model_id, device_map=None)
                tokenizer = processor.tokenizer


            elif 'wavelm' in args.model_name:
                from transformers import WavLMForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
                if 'KF' in model_id:
                    processor = Wav2Vec2Processor.from_pretrained(model_id)
                    tokenizer = processor.tokenizer
                    model = WavLMForCTC.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer), device_map=device_map)
                else:
                    from score_func import VALID_IPA_CHARS
                    phonemes = sorted(VALID_IPA_CHARS)
                    logger.info('phonemes:%s', phonemes)
                    vocab = ["<pad>", "<s>", "</s>", "<unk>"] + phonemes
                    vocab_dict = {token: idx for idx, token in enumerate(vocab)}

                    with open("/tmp/wavelm_vocab.json", "w", encoding="utf-8") as f:
                        json.dump(vocab_dict, f, ensure_ascii=False)
                    tokenizer = Wav2Vec2CTCTokenizer(
                        vocab_file='/tmp/wavelm_vocab.json',
                        unk_token="<unk>",
                        pad_token="<pad>",
                        word_delimiter_token=" ",
                        do_lower_case=False,
                    )

                    #feature_extractor = Wav2Vec2FeatureExtractor(
                    #    sampling_rate=16000,
                    #    return_attention_mask=True,
                    #)
                    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

                    processor = Wav2Vec2Processor(
                        feature_extractor=feature_extractor,
                        tokenizer=tokenizer,
                    )
                    kwargs = dict()
                    model = WavLMForCTC.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), ctc_loss_reduction=args.ctc_loss_reduction, pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer), mask_time_prob=args.mask_time_prob)   # important for new vocab
                    if args.frozen_audio:
                        model.freeze_feature_encoder()
                print(model.config)
            else:
                model_cls = AutoModelForSpeechSeq2Seq
                model = model_cls.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), trust_remote_code=True, **kwargs)
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                tokenizer = processor.tokenizer
        elif args.model_name.startswith('S2S'):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), trust_remote_code=True, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), trust_remote_code=True, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = model.cuda()

    logger.info('pad token id:%s, %s, padding side:%s', tokenizer.pad_token_id, model.config.pad_token_id, tokenizer.padding_side)

    for k in ["mask_feature_length", "mask_feature_min_masks", "mask_feature_prob", "mask_time_length", "mask_time_min_masks", "mask_time_prob"]:
        setattr(model, k, getattr(args, k))


    return model, tokenizer, processor


def load_model_for_predict(args, model_id):
    if torch.cuda.is_available():
        device_map = 'cuda'
    else:
        device_map = None
    if args.use_unsloth:
        model, tokenizer = load_unsloth_model(args, model_id)
    else:
        torch_dtype = getattr(torch, args.torch_dtype)
        if args.model_name.startswith('CSR'):
            if "qw3asr" in args.model_name:
                from qwen_asr import Qwen3ASRModel
                kwargs = dict()
                if args.use_flash2:
                    kwargs['attn_implementation'] = "flash_attention_2"
                asr_wrapper = Qwen3ASRModel.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), device_map=device_map, **kwargs)
                model = asr_wrapper.model
                #print(model.model.layers[0].self_attn)
                print(222, model.config._attn_implementation)
                model.generation_config = GenerationConfig.from_model_config(model.config)
                logger.info('use cache:%s', model.generation_config.use_cache)
                util.patch_outer_forward(model)
                processor = asr_wrapper.processor
                tokenizer = processor.tokenizer
            elif 'trans' in args.model_name:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), device_map=device_map)
                print(111111111, model_id)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                processor = tokenizer
            elif 'cohere' in args.model_name:
                from transformers import AutoProcessor, CohereAsrForConditionalGeneration
                processor = AutoProcessor.from_pretrained(model_id)
                model = CustomCohereAsrForConditionalGeneration.from_pretrained(model_id, device_map=device_map)
                tokenizer = processor.tokenizer
            elif 'wavelm' in args.model_name:
                from transformers import WavLMForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
                processor = Wav2Vec2Processor.from_pretrained(model_id)
                tokenizer = processor.tokenizer
                model = WavLMForCTC.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer), device_map=device_map)
            else:
                model_cls = AutoModelForSpeechSeq2Seq
                model = model_cls.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), trust_remote_code=True, device_map=device_map)
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                tokenizer = processor.tokenizer
                tokenizer.padding_side = 'left'
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side='right')
            if args.is_classify:
                model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch_dtype, num_labels=args.n_label, low_cpu_mem_usage=True, device_map="cuda")
            elif args.model_name.startswith('S2S'):
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=device_map)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=device_map)
        #util.restore_lm_head(args, model_id, model)
    if args.model_name.startswith('AR') or 'trans' in args.model_name:
        tokenizer.padding_side = 'left'
    logger.info('tokenizer padding side:%s, %s, %s', tokenizer.padding_side, tokenizer.pad_token_id, model.config.pad_token_id)

    logger.info('num of params for %s is %s', model_id, util.get_num_of_paras(model))
    model = model.eval()
    if args.compile_model:
        model.thinker = torch.compile(model.thinker, mode="reduce-overhead")
        logger.info('model compiled')
    return model, tokenizer, processor


def setup_training(args, model, processor, train_dataset, val_dataset, output_dir):
    training_args = TrainingArguments(
        remove_unused_columns=False,
        output_dir=output_dir,
        seed=args.seed,
        num_train_epochs=args.epochs,
        max_steps=-1,
        dataloader_num_workers=args.n_dl_worker,
        torch_compile=args.compile_model,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        #optim='adamw_torch',
        optim=args.optim,
        optim_target_modules=args.optim_target_modules if 'apollo_adamw' not in args.optim else [r".*.attn.*", r".*.mlp.*"],
        use_sampler=args.use_sampler,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        lr_scheduler_kwargs=args.lr_scheduler_paras,
        warmup_ratio=args.lr_warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_dir=output_dir,
        logging_steps=args.verbose,
        report_to=args.report_to,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        eval_delay=args.eval_delay,
        save_strategy=args.save_strategy,
        save_total_limit=args.n_keep_save,
        save_steps=args.eval_steps,
        save_only_model=not args.save_opt,
        load_best_model_at_end=True if args.do_val else False,
        bf16=args.mixed_precision=='bf16',
        fp16=args.mixed_precision=='fp16',
        do_train=args.do_train,
        do_eval=args.do_val,
        do_predict=args.predict_val,
        metric_for_best_model='loss',
        disable_tqdm=True,

        ##
        hard_ratio=args.hard_ratio,
        use_adam_mini=args.use_adam_mini,
        use_muon=args.use_muon,
        is_classify=args.is_classify,
        save_lm_head=args.save_lm_head,
        model_name=args.model_name,
        use_soft1=args.use_soft1,
        rdrop=args.rdrop,
        ema=args.ema,
    )

    # TRAIN
    cls = Trainer
    logger.info('cls for trainer:%s', cls)
    trainer = cls(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.collate,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience > 0 and args.do_val else None,

    )
    return trainer


def get_kf_data(args, data):
    if args.groupfy_col is not None:
        if args.stratify_col is not None:
            kf = StratifiedGroupKFold(n_splits=args.kn, shuffle=True, random_state=args.data_seed)
            splits = kf.split(data, y=data[args.stratify_col], groups=data[args.groupfy_col])
            for i in range(args.kn):
                train_inds, val_inds = next(splits)
                if i == args.kfid:
                    break
            train_data = data.iloc[train_inds]
            val_data = data.iloc[val_inds]
        else:
            kf = KFold(n_splits=args.kn, shuffle=True, random_state=args.data_seed)
            gps = np.array(sorted(data[args.groupfy_col].unique()))
            splits = kf.split(gps)
            for i in range(args.kn):
                train_inds, val_inds = next(splits)
                if i == args.kfid:
                    break
            train_gp = gps[train_inds]
            val_gp = gps[val_inds]
            train_data = data[data[args.group_col].isin(train_gp)]
            val_data = data[data[args.group_col].isin(val_gp)]
    else:
        if args.stratify_col is not None:
            kf = StratifiedKFold(n_splits=args.kn, shuffle=True, random_state=args.data_seed)
            splits = kf.split(data, data[args.stratify_col])
        else:
            kf = KFold(n_splits=args.kn, shuffle=True, random_state=args.data_seed)
            splits = kf.split(data, data.src)
        for i in range(args.kn):
            train_inds, val_inds = next(splits)
            if i == args.kfid:
                break
        train_data = data.iloc[train_inds]
        val_data = data.iloc[val_inds]
    train_data = train_data[:args.train_num]
    val_data = val_data[:args.val_num]
    return train_data, val_data

def prepare_dataset(args, **kwargs):
    train_data, val_data, test_data, train_ds, val_ds, test_ds = None, None, None, None, None, None
    if args.do_train or args.do_val:
        data = util.load_data(args)
        train_data, val_data = get_kf_data(args, data)

        if args.no_val:
            train_data = pd.concat([train_data, val_data])
            val_data = val_data[:100]
        if args.use_semi:
            train_data['split_id'] = 1000000
            semi_data = util.load_dump(args.semi_fpath)
            semi_data = semi_data[train_data.columns]
            _args = deepcopy(args)
            _args.data_seed = _args.semi_data_seed or _args.data_seed
            semi_train, semi_val = get_kf_data(_args, semi_data)
            if args.model_name.startswith('CSRW'):
                semi_train = pd.concat([semi_train, semi_val])
            logger.info('semi train:%s, semi_val:%s', len(semi_train), len(semi_val))
            train_data = pd.concat([train_data, semi_train])
            if args.predict_semi:
                val_data = semi_val
                val_data = val_data.sort_values(['audio_duration_sec', 'split_id'], ascending=False)
                logger.info('val semi:%s', len(val_data))
        #if args.use_semi:
        #    semi_args = deepcopy(args)
        #    semi_args.dataset = 'semi'
        #    semi = util.load_data(semi_args)
        #    train_data = pd.concat([semi, train_data])

        if args.val_smoke:
            if args.model_name.startswith('CSRW'):
                smoke_data = pd.DataFrame(util.load_json_lines('../data/csrw/submission_format_z2HCh3r.jsonl'))
            else:
                smoke_data = pd.DataFrame(util.load_json_lines('../data/csrp/submission_format_5UPXd8x.jsonl'))
            val_data = data[data.utterance_id.isin(smoke_data.utterance_id)]
            train_data = data[~data.utterance_id.isin(val_data.utterance_id)]
            train_data = train_data[train_data.audio_duration_sec < (args.max_sec or 1000000)]
            if not args.is_eval:
                val_data = val_data[val_data.audio_duration_sec < (args.max_sec or 1000000)]
        if args.eval_long and args.dataset in ['csrw', 'csrp']:
            args2 = deepcopy(args)
            args2.max_sec = 10000
            df2 = util.load_data(args2)
            df2 = df2[~df2.utterance_id.isin(data.utterance_id)]
            val_data = pd.concat([val_data, df2])
        train_ds = gen_ds(args, 'train', train_data, **kwargs)
        val_ds = gen_ds(args, 'val', val_data, **kwargs)
        logger.info('train ds:%s, val_ds:%s', len(train_ds), len(val_ds))

    if args.do_test:
        test_args = deepcopy(args)
        test_args.data_type = 'test'
        test_data = util.load_data(test_args)
        if args.split_id>=0:
            num = len(test_data)
            if args.split_id==0:
                test_data = test_data[:num//2]
            else:
                assert args.split_id == 1
                test_data = test_data[num // 2:]
        test_ds = gen_ds(args, 'test', test_data, **kwargs)
    return train_data, val_data, test_data, train_ds, val_ds, test_ds

def pool(hidden_states, attention_mask):
    token_indices = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.int32)
    last_non_pad_token = (token_indices * attention_mask).argmax(-1)
    hidden_states = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), last_non_pad_token]
    return hidden_states


def eval_vllm(args, model, prompts, lora_request=None):
    from vllm import SamplingParams
    sp = SamplingParams(seed=args.seed, temperature=args.temp, skip_special_tokens=True, max_tokens=args.max_gen_len, top_p=args.topp)
    outputs = model.generate(prompts, sp, use_tqdm=True, lora_request=lora_request)
    outputs = [output.outputs[0].text for output in outputs]
    logger.info('vllm outputs:%s', '-----------\n'.join(outputs[:2]))
    return outputs


def get_vllm_inputs(args, ds, tokenizer):
    row_ids, prompts, labels = [], [], []
    for i, batch in tqdm(enumerate(ds), total=len(ds)):
        for item in batch:
            row_ids.append(item['row_id'])
            prompts.append(item['input_text'])
            if 'orig_label' in item:
                labels.append(item['orig_label'])
    logger.info('prompts:%s', prompts[:2])
    df = pd.DataFrame(dict(row_id=row_ids, prompt=prompts))
    if len(labels)>0:
        df['label'] = labels
    return df


def eval_vllm_ar(args, dl, model, lora_request, tokenizer, output_dir):
    preds = get_vllm_inputs(args, dl, tokenizer)
    outputs = eval_vllm(args, model, preds.prompt.values, lora_request=lora_request)
    preds['pred'] = outputs
    return preds

def eval_ar(args, model, lora_request, tokenizer, output_dir):
    train_data, val_data, test_data, train_ds, val_ds, test_ds = prepare_dataset(args, tokenizer=tokenizer)
    if args.data_type != 'test':
        if args.use_batch_sampler:
            batch_sampler = DynamicBatchSampler(val_ds)
            dl = torch.utils.data.DataLoader(val_ds, batch_sampler=batch_sampler, pin_memory=True, num_workers=args.n_dl_worker,
                                             collate_fn=val_ds.collate)
        else:
            dl = torch.utils.data.DataLoader(val_ds, batch_size=args.val_batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=val_ds.collate)
    else:
        if args.use_batch_sampler:
            batch_sampler = DynamicBatchSampler(test_ds)
            dl = torch.utils.data.DataLoader(test_ds, batch_sampler=batch_sampler, pin_memory=True, num_workers=args.n_dl_worker,
                                         collate_fn=test_ds.collate)
        else:
            dl = torch.utils.data.DataLoader(test_ds, batch_size=args.val_batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=test_ds.collate)
    if args.use_vllm:
        preds = eval_vllm_ar(args, dl, model, lora_request, tokenizer, output_dir)
    else:
        with torch.no_grad():
            rsts = []
            for batch in tqdm(dl, desc='eval_ar'):
                input_ids, attention_mask = batch['input_ids'].cuda(), batch['attention_mask'].cuda()
                generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=args.max_gen_len,
                                               temperature=args.temp,
                                               num_beams=args.n_beam,
                                               num_return_sequences=1,
                                               early_stopping=True,
                                               do_sample=False,
                                               )
                for gen_ids, input_ids, row_id in zip(generated_ids, input_ids, batch['row_id']):
                    if args.model_name.startswith('S2S'):
                        output_ids = gen_ids.tolist()
                    else:
                        output_ids = gen_ids[len(input_ids):].tolist()
                    output = tokenizer.decode(output_ids, skip_special_tokens=True)
                    rsts.append([row_id, output])
        preds = pd.DataFrame(rsts, columns=['row_id', 'pred'])

    fpath = f"{output_dir}/pred{args.suffix}_{args.data_type}.dump"
    util.dump(preds, fpath)
    logger.info('preds saved to %s', fpath)
    return preds, val_data


def eval_trans(args, model, lora_request, tokenizer, processor, output_dir):
    train_data, val_data, test_data, train_ds, val_ds, test_ds = prepare_dataset(args, tokenizer=tokenizer, processor=processor)
    ds_cls = args.test_ds_cls if args.data_type == 'test' else args.val_ds_cls
    if 'Dynamic' in ds_cls:
        batch_size = None
    else:
        batch_size = args.val_batch_size
    if args.data_type != 'test':
        dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=val_ds.collate)
    else:
        dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=test_ds.collate)
    if args.use_vllm:
        preds = eval_vllm_ar(args, dl, model, lora_request, tokenizer, output_dir)
    else:
        has_cuda = torch.cuda.is_available()
        # if 1==1:
        with torch.no_grad():
            rsts = []
            logger.info('total batch:%s', len(dl))
            if args.dynamic_beam > 0:
                num_thr = int(len(dl) * args.dynamic_beam)
            num = 0
            for batch in tqdm(dl, desc='eval_trans'):
                # for batch in tqdm(dl, desc='eval_csr'):
                num += 1
                if args.dynamic_beam > 0 and num < num_thr:
                    beam_size = 1
                    temp = 1
                else:
                    beam_size = args.n_beam
                    temp = args.temp
                utterance_ids = batch.pop('utterance_id')
                durations = batch.pop('duration')
                split_ids = batch.pop('split_id', [0] * len(utterance_ids))
                _ = batch.pop('seq_len')
                for k in batch:
                    if has_cuda:
                        batch[k] = batch[k].cuda()
                max_new_tokens = args.max_gen_len
                # max_new_tokens = int(args.max_gen_len*(max(durations)+2)/args.max_sec)
                # max_new_tokens = max(32, int(args.max_gen_len*max(durations)/args.max_sec))
                text_ids = model.generate(**batch, max_new_tokens=max_new_tokens,
                                          temperature=temp,
                                          num_beams=beam_size,
                                          num_return_sequences=1,
                                          early_stopping=True,
                                          do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id,
                                          use_cache=True,
                                          )
                # print(111, tokenizer.batch_decode(batch['input_ids']))
                generate_texts = tokenizer.batch_decode(
                    text_ids[:, batch["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                # print(222, generate_texts)
                # exit()
                for generate_text, utterance_id, split_id in zip(generate_texts, utterance_ids, split_ids):
                    rsts.append([utterance_id, generate_text, split_id])
        preds = pd.DataFrame(rsts, columns=['utterance_id', 'pred', 'split_id'])
        if args.no_merge:
            preds = pd.DataFrame(rsts, columns=['utterance_id', 'pred', 'split_id'])
        else:
            preds = preds.sort_values(['utterance_id', 'split_id']).groupby('utterance_id', as_index=False).agg(pred=('pred', ''.join))

    fpath = f"{output_dir}/pred{args.suffix}_{args.data_type}.dump"
    util.dump(preds, fpath)
    logger.info('preds saved to %s', fpath)

    return preds, val_data

def eval_csrp(args, model, lora_request, tokenizer, processor, output_dir):
    train_data, val_data, test_data, train_ds, val_ds, test_ds = prepare_dataset(args, tokenizer=tokenizer, processor=processor)
    ds_cls = args.test_ds_cls if args.data_type == 'test' else args.val_ds_cls
    if 'Dynamic' in ds_cls:
        batch_size = None
    else:
        batch_size = args.val_batch_size
    if args.data_type != 'test':
        dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=val_ds.collate)
    else:
        dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=test_ds.collate)
    if args.use_vllm:
        preds = eval_vllm_ar(args, dl, model, lora_request, tokenizer, output_dir)
    else:
        has_cuda = torch.cuda.is_available()
        # if 1==1:
        with torch.no_grad():
            rsts = []
            logger.info('total batch:%s', len(dl))
            if args.dynamic_beam > 0:
                num_thr = int(len(dl) * args.dynamic_beam)
            num = 0
            for batch in tqdm(dl, desc="eval_csrp"):
                # for batch in tqdm(dl, desc='eval_csr'):
                num += 1
                if args.dynamic_beam > 0 and num < num_thr:
                    beam_size = 1
                    temp = 1
                else:
                    beam_size = args.n_beam
                    temp = args.temp
                utterance_ids = batch.pop('utterance_id')
                durations = batch.pop('duration')
                split_ids = batch.pop('split_id', [0] * len(utterance_ids))
                for k in batch:
                    if has_cuda:
                        batch[k] = batch[k].cuda()
                    if k in ['input_features']:
                        batch[k] = batch[k].to(model.dtype)
                max_new_tokens = args.max_gen_len
                # max_new_tokens = int(args.max_gen_len*(max(durations)+2)/args.max_sec)
                # max_new_tokens = max(32, int(args.max_gen_len*max(durations)/args.max_sec))
                text_ids = model.generate(**batch, max_new_tokens=max_new_tokens,
                                          temperature=temp,
                                          num_beams=beam_size,
                                          num_return_sequences=1,
                                          early_stopping=True,
                                          do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id,
                                          use_cache=True,
                                          )
                # print(111, tokenizer.batch_decode(batch['input_ids']))
                generate_texts = processor.batch_decode(
                    text_ids.sequences[:, batch["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                # print(222, generate_texts)
                # exit()
                for generate_text, utterance_id, split_id, duration in zip(generate_texts, utterance_ids, split_ids, durations):
                    rsts.append([utterance_id, generate_text, split_id, duration])
        preds = pd.DataFrame(rsts, columns=['utterance_id', 'pred', 'split_id', 'audio_duration_sec'])
        if args.no_merge:
            preds = pd.DataFrame(rsts, columns=['utterance_id', 'pred', 'split_id'])
        else:
            preds = preds.sort_values(['utterance_id', 'split_id']).groupby('utterance_id', as_index=False).agg(pred=('pred', ''.join))

    fpath = f"{output_dir}/pred{args.suffix}_{args.data_type}.dump"
    util.dump(preds, fpath)
    logger.info('preds saved to %s', fpath)

    return preds, val_data

def eval_csr(args, model, lora_request, tokenizer, processor, output_dir):
    if 'trans' in args.model_name:
        return eval_trans(args, model, lora_request, tokenizer, processor, output_dir)
    if 'CSRP' in args.model_name:
        return eval_csrp(args, model, lora_request, tokenizer, processor, output_dir)
    if 'granite' in args.model_name:
        return eval_granite(args, model, lora_request, tokenizer, processor, output_dir)
    if 'wavelm' in args.model_name:
        return eval_wavelm(args, model, lora_request, tokenizer, processor, output_dir)
    if 'cohere' in args.model_name:
        return eval_cohere(args, model, lora_request, tokenizer, processor, output_dir)
    train_data, val_data, test_data, train_ds, val_ds, test_ds = prepare_dataset(args, tokenizer=tokenizer, processor=processor)
    ds_cls = args.test_ds_cls if args.data_type=='test' else args.val_ds_cls
    if 'Dynamic' in ds_cls:
        batch_size = None
    else:
        batch_size = args.val_batch_size
    if args.data_type != 'test':
        dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=val_ds.collate)
    else:
        dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=test_ds.collate)
    if args.use_vllm:
        preds = eval_vllm_ar(args, dl, model, lora_request, tokenizer, output_dir)
    else:
        has_cuda = torch.cuda.is_available()
        #if 1==1:
        with torch.no_grad():
            rsts = []
            logger.info('total batch:%s', len(dl))
            if args.dynamic_beam>0:
                num_thr = int(len(dl)*args.dynamic_beam)
            num = 0
            #for batch in dl:
            for batch in tqdm(dl, desc='eval_csr'):
                num += 1
                if args.dynamic_beam>0 and num<num_thr:
                    beam_size = 1
                    temp = 1
                else:
                    beam_size = args.n_beam
                    temp = args.temp
                utterance_ids = batch.pop('utterance_id')
                durations = batch.pop('duration')
                split_ids = batch.pop('split_id', [0]*len(utterance_ids))
                for k in batch:
                    if has_cuda:
                        batch[k] = batch[k].cuda()
                    if k in ['input_features']:
                        batch[k] = batch[k].to(model.dtype)
                max_new_tokens = args.max_gen_len
                #max_new_tokens = int(args.max_gen_len*(max(durations)+2)/args.max_sec)
                #max_new_tokens = max(32, int(args.max_gen_len*max(durations)/args.max_sec))
                text_ids = model.generate(**batch, max_new_tokens=max_new_tokens,
                                          temperature=temp,
                                          num_beams=beam_size,
                                          num_return_sequences=1,
                                          early_stopping=True,
                                          do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id,
                                          use_cache=True,
                                          )
                #print(111, tokenizer.batch_decode(batch['input_ids']))
                generate_texts = processor.batch_decode(
                    text_ids.sequences[:, batch["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                #print(222, generate_texts)
                #exit()
                for generate_text, utterance_id, split_id in zip(generate_texts, utterance_ids, split_ids):
                    rsts.append([utterance_id, generate_text, split_id])
        preds = pd.DataFrame(rsts, columns=['utterance_id', 'pred', 'split_id'])
        if args.no_merge:
            preds = pd.DataFrame(rsts, columns=['utterance_id', 'pred', 'split_id'])
        else:
            preds = preds.sort_values(['utterance_id', 'split_id']).groupby('utterance_id', as_index=False).agg(pred=('pred', ''.join))

    fpath = f"{output_dir}/pred{args.suffix}_{args.data_type}.dump"
    util.dump(preds, fpath)
    logger.info('preds saved to %s', fpath)

    return preds, val_data

def eval_cohere(args, model, lora_request, tokenizer, processor, output_dir):
    train_data, val_data, test_data, train_ds, val_ds, test_ds = prepare_dataset(args, tokenizer=tokenizer, processor=processor)
    ds_cls = args.test_ds_cls if args.data_type=='test' else args.val_ds_cls
    if 'Dynamic' in ds_cls:
        batch_size = None
    else:
        batch_size = args.val_batch_size
    if args.data_type != 'test':
        dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=val_ds.collate)
    else:
        dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=test_ds.collate)
    if args.use_vllm:
        preds = eval_vllm_ar(args, dl, model, lora_request, tokenizer, output_dir)
    else:
        has_cuda = torch.cuda.is_available()
        #if 1==1:
        with torch.no_grad():
            rsts = []
            logger.info('total batch:%s', len(dl))
            if args.dynamic_beam>0:
                num_thr = int(len(dl)*args.dynamic_beam)
            num = 0
            #for batch in tqdm(dl, desc='eval_csr'):
            for batch in dl:
                num += 1
                if args.dynamic_beam>0 and num<num_thr:
                    beam_size = 1
                    temp = 1
                else:
                    beam_size = args.n_beam
                    temp = args.temp
                utterance_ids = batch.pop('utterance_id')
                durations = batch.pop('duration')
                split_ids = batch.pop('split_id', [0]*len(utterance_ids))
                _ = batch.pop('audio_chunk_index')
                for k in batch:
                    if has_cuda:
                        batch[k] = batch[k].cuda()
                    if k in ['input_features']:
                        batch[k] = batch[k].to(model.dtype)
                max_new_tokens = args.max_gen_len
                #max_new_tokens = int(args.max_gen_len*(max(durations)+2)/args.max_sec)
                #max_new_tokens = max(32, int(args.max_gen_len*max(durations)/args.max_sec))
                text_ids = model.generate(**batch, max_new_tokens=max_new_tokens,
                                          temperature=temp,
                                          num_beams=beam_size,
                                          num_return_sequences=1,
                                          early_stopping=True,
                                          do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id,
                                          use_cache=True,
                                          )
                #print(111, tokenizer.batch_decode(batch['input_ids']))
                generate_texts = processor.batch_decode(
                    text_ids,
                    skip_special_tokens=True,
                )
                #print(222, generate_texts)
                #exit()
                for generate_text, utterance_id, split_id in zip(generate_texts, utterance_ids, split_ids):
                    rsts.append([utterance_id, generate_text, split_id])
        preds = pd.DataFrame(rsts, columns=['utterance_id', 'pred', 'split_id'])
        preds = preds.sort_values(['utterance_id', 'split_id']).groupby('utterance_id', as_index=False).agg(pred=('pred', ''.join))

    fpath = f"{output_dir}/pred{args.suffix}_{args.data_type}.dump"
    util.dump(preds, fpath)
    logger.info('preds saved to %s', fpath)

    return preds, val_data

def eval_wavelm(args, model, lora_request, tokenizer, processor, output_dir):
    train_data, val_data, test_data, train_ds, val_ds, test_ds = prepare_dataset(args, tokenizer=tokenizer, processor=processor)
    ds_cls = args.test_ds_cls if args.data_type=='test' else args.val_ds_cls
    if args.n_beam>1:
        from pyctcdecode import build_ctcdecoder

        # Get the list of characters (labels) from your processor
        # Ensure you sort them by their index
        labels = [label for label, idx in sorted(processor.tokenizer.get_vocab().items(), key=lambda x: x[1])]

        # Build the decoder
        # If you don't have a Language Model (KenLM), pass None
        decoder = build_ctcdecoder(
            labels,
            word_delimiter_token=" ",
            kenlm_model_path=None,  # Path to .arpa or .bin file if you have one
        )

    if 'Dynamic' in ds_cls:
        batch_size = None
    else:
        batch_size = args.val_batch_size
    if args.data_type != 'test':
        dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=val_ds.collate)
    else:
        dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=test_ds.collate)
    if args.use_vllm:
        preds = eval_vllm_ar(args, dl, model, lora_request, tokenizer, output_dir)
    else:
        has_cuda = torch.cuda.is_available()
        #if 1==1:
        with torch.no_grad():
            rsts = []
            logger.info('total batch:%s', len(dl))
            if args.dynamic_beam>0:
                num_thr = int(len(dl)*args.dynamic_beam)
            num = 0
            for batch in tqdm(dl, desc='eval_csr'):
                num += 1
                if args.dynamic_beam>0 and num<num_thr:
                    beam_size = 1
                else:
                    beam_size = args.n_beam
                utterance_ids = batch.pop('utterance_id')
                durations = batch.pop('duration')
                split_ids = batch.pop('split_id', [0]*len(utterance_ids))
                for k in batch:
                    if has_cuda:
                        batch[k] = batch[k].cuda()
                    if k in ['input_values']:
                        batch[k] = batch[k].to(model.dtype)
                max_new_tokens = args.max_gen_len
                #max_new_tokens = int(args.max_gen_len*(max(durations)+2)/args.max_sec)
                #max_new_tokens = max(32, int(args.max_gen_len*max(durations)/args.max_sec))
                logits = model(**batch).logits
                if beam_size>1:
                    logits_np = logits.cpu().numpy()
                    # Perform Beam Search
                    generate_texts = []
                    for logit in logits_np:
                        beam_search_output = decoder.decode(logit, beam_width=beam_size)
                        generate_texts.append(beam_search_output)
                else:
                    #print(111, logits.shape)
                    #probs = torch.nn.functional.softmax(logits, dim=-1)
                    #print(222, probs[..., 0].mean())
                    #max_probs, predicted_ids = torch.max(probs, dim=-1)
                    #print("Average Max Probability:", max_probs.mean().item())
                    #print("Unique Predicted IDs:", torch.unique(predicted_ids))

                    predicted_ids = torch.argmax(logits, dim=-1)
                    generate_texts = processor.batch_decode(predicted_ids)
                    #print(111, generate_texts)
                    #exit()

                for generate_text, utterance_id, split_id in zip(generate_texts, utterance_ids, split_ids):
                    rsts.append([utterance_id, generate_text, split_id])
        preds = pd.DataFrame(rsts, columns=['utterance_id', 'pred', 'split_id'])
        preds = preds.sort_values(['utterance_id', 'split_id']).groupby('utterance_id', as_index=False).agg(pred=('pred', ''.join))

    fpath = f"{output_dir}/pred{args.suffix}_{args.data_type}.dump"
    util.dump(preds, fpath)
    logger.info('preds saved to %s', fpath)

    return preds, val_data
def eval_granite(args, model, lora_request, tokenizer, processor, output_dir):
    train_data, val_data, test_data, train_ds, val_ds, test_ds = prepare_dataset(args, tokenizer=tokenizer, processor=processor)
    ds_cls = args.test_ds_cls if args.data_type=='test' else args.val_ds_cls
    if 'Dynamic' in ds_cls:
        batch_size = None
    else:
        batch_size = args.val_batch_size
    if args.data_type != 'test':
        dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=val_ds.collate)
    else:
        dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=test_ds.collate)
    if args.use_vllm:
        preds = eval_vllm_ar(args, dl, model, lora_request, tokenizer, output_dir)
    else:
        has_cuda = torch.cuda.is_available()
        #if 1==1:
        with torch.no_grad():
            rsts = []
            logger.info('total batch:%s', len(dl))
            if args.dynamic_beam>0:
                num_thr = int(len(dl)*args.dynamic_beam)
            num = 0
            for batch in tqdm(dl, desc='eval_csr'):
                num += 1
                if args.dynamic_beam>0 and num<num_thr:
                    beam_size = 1
                else:
                    beam_size = args.n_beam
                utterance_ids = batch.pop('utterance_id')
                durations = batch.pop('duration')
                split_ids = batch.pop('split_id', [0]*len(utterance_ids))
                for k in batch:
                    if has_cuda:
                        batch[k] = batch[k].cuda()
                    if k in ['input_features']:
                        batch[k] = batch[k].to(model.dtype)
                max_new_tokens = args.max_gen_len
                #max_new_tokens = int(args.max_gen_len*(max(durations)+2)/args.max_sec)
                #max_new_tokens = max(32, int(args.max_gen_len*max(durations)/args.max_sec))
                text_ids = model.generate(**batch, max_new_tokens=max_new_tokens,
                                               temperature=args.temp,
                                               num_beams=beam_size,
                                               num_return_sequences=1,
                                               early_stopping=True,
                                               do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id,
                                               use_cache=True,
                                               )
                #print(111, tokenizer.batch_decode(batch['input_ids']))
                generate_texts = processor.batch_decode(
                    text_ids[:, batch["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                #print(222, generate_texts)
                #exit()
                for generate_text, utterance_id, split_id in zip(generate_texts, utterance_ids, split_ids):
                    rsts.append([utterance_id, generate_text, split_id])
        preds = pd.DataFrame(rsts, columns=['utterance_id', 'pred', 'split_id'])
        preds = preds.sort_values(['utterance_id', 'split_id']).groupby('utterance_id', as_index=False).agg(pred=('pred', ''.join))

    fpath = f"{output_dir}/pred{args.suffix}_{args.data_type}.dump"
    util.dump(preds, fpath)
    logger.info('preds saved to %s', fpath)

    return preds, val_data

def predict_zipa(args):
    from zipa_transducer_inference import initialize_model
    output_dir = f"{args.data_dir}/{args.model_name}_KF{args.kfid}"
    model = initialize_model(f"{output_dir}/best-valid-loss.pt", '../data/zipa/unigram_127.model')
    train_data, val_data, test_data, train_ds, val_ds, test_ds = prepare_dataset(args, tokenizer=None, processor=None)
    ds_cls = args.test_ds_cls if args.data_type == 'test' else args.val_ds_cls
    if 'Dynamic' in ds_cls:
        batch_size = None
    else:
        batch_size = args.val_batch_size
    if args.data_type != 'test':
        dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=val_ds.collate)
    else:
        dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                         shuffle=False, drop_last=False, collate_fn=test_ds.collate)
    has_cuda = torch.cuda.is_available()
    #if 1==1:
    with torch.no_grad():
        rsts = []
        logger.info('total batch:%s', len(dl))
        for batch in tqdm(dl, desc='eval_csr'):
            utterance_ids = batch.pop('utterance_id')
            durations = batch.pop('duration')
            split_ids = batch.pop('split_id', [0]*len(utterance_ids))
            for k in batch:
                if has_cuda:
                    batch[k] = batch[k].cuda()
            generate_texts = model.predict(**batch)
            for generate_text, utterance_id, split_id in zip(generate_texts, utterance_ids, split_ids):
                rsts.append([utterance_id, ' '.join(generate_text), split_id])
    preds = pd.DataFrame(rsts, columns=['utterance_id', 'pred', 'split_id'])
    preds = preds.sort_values(['utterance_id', 'split_id']).groupby('utterance_id', as_index=False).agg(pred=('pred', ''.join))

    fpath = f"{output_dir}/pred{args.suffix}_{args.data_type}.dump"
    util.dump(preds, fpath)
    logger.info('preds saved to %s', fpath)

    return preds, val_data

def predict(args):
    if 'zipa' in args.model_name:
        return predict_zipa(args)
    logger.info("model:%s", args.model_name)
    args.is_eval = True
    output_dir = f"{args.data_dir}/{args.model_name}_KF{args.kfid}"
    if args.restore:
        util.restore_args(args, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if args.restore:
        if args.restore_step is not None:
            ckpt_dir = f"{output_dir}/checkpoint-{args.restore_step}"
        else:
            ckpt_dir = sorted(glob(f"{output_dir}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))[-1]
    else:
        ckpt_dir = args.backbone
    logger.info('restore from ckpt:%s', ckpt_dir)

    if args.use_vllm:
        model, tokenizer, lora_request = util.get_vllm(args, ckpt_dir, max_num_seqs=args.max_num_seqs)
    else:
        model, tokenizer, processor = load_model_for_predict(args, ckpt_dir)
        lora_request = None
    if args.model_name.startswith('CSR'):
        return eval_csr(args, model, lora_request, tokenizer, processor, output_dir)
    elif args.model_name.startswith('AR'):
        return eval_ar(args, model, lora_request, tokenizer, output_dir)
    elif args.model_name.startswith('S2S'):
        return eval_ar(args, model, lora_request, tokenizer, output_dir)


def create_lora_ms(args):
    from safetensors.torch import  load_file, save_file
    model_names = args.model_name.split()
    for i, model_name in enumerate(model_names):
        print(i, model_name)
        ckpt_dir = util.get_ckpt(f"{args.data_dir}/{model_name}", restore_step=args.restore_step)
        if i==0:
            out_dir = ckpt_dir.split('_KF')[0] + '_ms_KF0'
            #if os.path.exists(out_dir):
            #os.makedirs(out_dir, exist_ok=True)
            state_dict = load_file(f"{ckpt_dir}/adapter_model.safetensors")
            for k, v in state_dict.items():
                state_dict[k] = v/len(model_names)
        else:
            for k, v in load_file(f"{ckpt_dir}/adapter_model.safetensors").items():
                state_dict[k] += v/len(model_names)
    step = os.path.basename(ckpt_dir)
    shutil.copytree(f"{args.data_dir}/{model_name}", f"{out_dir}", ignore=shutil.ignore_patterns('checkpoint-*/adapter_model.safetensors'))
    save_file(state_dict, f"{out_dir}/{step}/adapter_model.safetensors")
    print('Done!')



def main(args):
    logger.info('backbone: %s, kfid: %s', args.backbone, args.kfid)
    set_seed(args.seed)

    model, tokenizer, processor = load_model(args, args.backbone)
    tokenizer.padding_side = 'right'
    output_dir = f"{args.output_dir}/{args.model_name}_KF{args.kfid}"
    os.makedirs(output_dir, exist_ok=True)
    util.dump_json(args.__dict__, f'{output_dir}/args.json')

    logger.info('num of params %s', util.get_num_of_paras(model))

    train_data, val_data, test_data, train_ds, val_ds, test_ds = prepare_dataset(args, tokenizer=tokenizer, model_config=model.config, processor=processor)
    trainer = setup_training(args, model, processor, train_ds, val_ds, output_dir)
    if args.do_train:
        trainer.train()
        logger.info('train DONE!')
    if not args.do_train and args.do_val:
        trainer.evaluate()
        logger.info('eval DONE!')
    if args.do_test:
        outputs = trainer.predict(test_ds)
        util.dump(outputs, f'{args.output_dir}/{args.model_name}/pred_test.dump')
        print(outputs.keys())
        logger.info('test DONE!')
    logger.info('DONE!')

if __name__ == "__main__":
    util.set_logger()
    if args.debug:
        args.backbone = 'HuggingFaceTB/SmolLM-135M'
        args.num_train_epochs = 2
        args.num = 1000000
        args.eval_steps = 10
        args.batch_size = 1
        args.val_batch_size = 1
        args.dataloader_num_workers = 0
        args.gradient_accumulation_steps = 1
        args.do_train = True
        args.seed = 9528
        args.kn = 2
        args.use_full = True
        args.disable_tqdm = True
        args.ds_cls = 'Dataset'
        args.val_ds_cls = 'Dataset'
        #args.max_seq_len = 8
        args.max_seq_len = 4096
        args.max_gen_len = 1024

    if args.method_name is not None and args.method_name.startswith('predict'):
        preds, val_datas = [], []
        for kfid in args.kfids.split():
            args.kfid = int(kfid)
            pred, val_data = globals()[args.method_name](args)
            preds.append(pred)
            val_datas.append(val_data)
        if args.scoring:
            preds = pd.concat(preds)
            vals = pd.concat(val_datas)
            s = util.score(preds, vals, is_word=args.model_name.startswith('CSRW'))
            logger.info('kf score:%s', s)
    elif args.method_name is not None:
        globals()[args.method_name](args)
    else:
        for kfid in args.kfids.split():
            my_args = deepcopy(args)
            my_args.kfid = int(kfid)
            my_args.seed = my_args.seed + my_args.kfid
            main(my_args)
            torch.cuda.empty_cache()
