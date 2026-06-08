import os, sys, json, logging
from copy import deepcopy
import pickle
import pandas as pd
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import time
from argparse import ArgumentParser
from multiprocessing import Pool
import numpy as np
import subprocess
from contextlib import contextmanager
import re
import hashlib

import torch
from torch.nn import functional as F
import numpy as np
import math
import pandas
import librosa
try:
    from csrw.score_func import score_wer, score_ipa_cer
except Exception as e:
    print('error import score func', e)
    pass

try:
    import eng_to_ipa as ipa
except:
    pass


from functools import partial

logger = logging.getLogger(__name__)


parser = ArgumentParser(conflict_handler='resolve')

parser.add_argument("-fix_continuous", action="store_true")
parser.add_argument("-predict_semi", action="store_true")
parser.add_argument("-no_merge", action="store_true")
parser.add_argument("-quantization")
parser.add_argument("-ppt")
parser.add_argument("-pred_fpath")
parser.add_argument("-audio_att_dropout", type=float, default=0)
parser.add_argument("-text_att_dropout", type=float, default=0)
parser.add_argument("-dynamic_power", type=float, default=0.4)
parser.add_argument("-dynamic_beam", type=float, default=0)
parser.add_argument("-nemo_model_path")
parser.add_argument("-semi_data_seed", type=int)
parser.add_argument("-max_sec", type=int)
parser.add_argument("-min_sec", type=int)
parser.add_argument("-max_sec2", type=int)
parser.add_argument("-aug_clone", type=float, default=0)
parser.add_argument("-eval_long", action="store_true")
parser.add_argument("-ctc_loss_reduction", default="sum")
parser.add_argument("-n_clone", type=int)
parser.add_argument("-gen_split", type=int, default=-1)
parser.add_argument("-use_flash2", action="store_true")
parser.add_argument("-use_gen_audio", action="store_true")
parser.add_argument("-use_tb", action="store_true", default=None)
parser.add_argument("-val_smoke", action="store_true")
parser.add_argument("-is_classify", action="store_true", default=None)
parser.add_argument("-frozen_audio", action="store_true", default=None)
parser.add_argument("-frozen_llm", action="store_true", default=None)
parser.add_argument("-frozen_encoder", action="store_true", default=None)
parser.add_argument("-use_adam_mini", action="store_true")
parser.add_argument("-use_muon", action="store_true")
parser.add_argument("-use_boundary", action="store_true")
parser.add_argument("-use_soft1", action="store_true")
parser.add_argument("-semi_fpath")
parser.add_argument("-gen_dir", default='gen_data')
parser.add_argument("-sr", type=int)
parser.add_argument("-mask_feature_prob", type=float, default=0)
parser.add_argument("-mask_feature_length", type=int, default=5)
parser.add_argument("-mask_feature_min_masks", type=int, default=0)
parser.add_argument("-mask_time_prob", type=float, default=0)
parser.add_argument("-mask_time_length", type=int, default=10)
parser.add_argument("-mask_time_min_masks", type=int, default=2)
parser.add_argument("-aug_cat", type=float, default=0)
# audioaugmentations
parser.add_argument("-am_gs", type=float, default=0)
parser.add_argument("-am_gs_db", type=float, default=4)
parser.add_argument("-am_bn_noise", type=float, default=0)
parser.add_argument("-am_bn_speech_db", type=float, default=10)
parser.add_argument("-am_bn_noise_db", type=float, default=3)
parser.add_argument("-am_bn_noise_max_db", type=float, default=30)

parser.add_argument("-ds", "--dataset", default='csrw')
parser.add_argument("-d", "--debug",  action="store_true")
parser.add_argument("-data_type", default='train')
parser.add_argument("-nv","--no_val", action="store_true")
parser.add_argument("-is_eval",  action="store_true")
parser.add_argument("-ds_cls")
parser.add_argument("-val_ds_cls")
parser.add_argument("-test_ds_cls")
parser.add_argument("-data_type", default='train')
parser.add_argument("-data_dir", default='../data')
parser.add_argument("-kn", type=int)
parser.add_argument("-kfid", type=int, default=0)
parser.add_argument("-kfids")
parser.add_argument("-groupfy_col")
parser.add_argument("-stratify_col")
parser.add_argument("-model_name")
parser.add_argument("-backbone")
parser.add_argument("-activation")
parser.add_argument("-n_repeat", type=int, default=1)
parser.add_argument("-n_layer", type=int)
parser.add_argument("-d_model", type=int)
parser.add_argument("-d_ffd", type=int)
parser.add_argument("-n_head", type=int)
parser.add_argument("-dropout", type=float, default=0)
parser.add_argument("-rdrop", type=float, default=0)
parser.add_argument("-val_pct", default=0.1)
parser.add_argument("-seed", type=int)
parser.add_argument("-data_seed", type=int)
parser.add_argument("-unsloth_seed", type=int)
parser.add_argument("-n_xtoken", type=int)
parser.add_argument("-prefix")
parser.add_argument("-max_seq_len", type=int)
parser.add_argument("-max_gen_len", type=int)
parser.add_argument("-temp", type=float, default=1.0)
parser.add_argument("-n_beam", type=int, default=1)
parser.add_argument("-do_sample",  action="store_true")
parser.add_argument("-mixed_precision", default='no')
parser.add_argument("-cpu",  action="store_true")
parser.add_argument("-cudnn_benchmark",  action="store_true")
parser.add_argument("-deterministic_algorithms",  action="store_true")
parser.add_argument("-save",  action="store_true")
parser.add_argument("-save_half",  action="store_true")
parser.add_argument("-save_state",  action="store_true")
parser.add_argument("-save_best",  action="store_true")
parser.add_argument("-ckpts", default=None)
parser.add_argument("-n_keep_save",  type=int, default=1)
parser.add_argument("-save_epoch",  type=int, default=100000000000)
parser.add_argument("-save_opt",  action="store_true")
parser.add_argument("-remove_unused_columns",  action="store_true")
parser.add_argument("-gradient_checkpointing",  action="store_true")
parser.add_argument("-use_pretrain",  action="store_true")
parser.add_argument("-use_sampler",  action="store_true")
parser.add_argument("-use_badam",  action="store_true")
parser.add_argument("-use_full",  action="store_true")
parser.add_argument("-switch_block_every",  type=int, default=32)
parser.add_argument("-use_score_scaling",  action="store_true")
parser.add_argument("-use_double_quant",  action="store_true")
parser.add_argument("-m", "--method_name")
parser.add_argument("-compile_model", action="store_true")
parser.add_argument("-compile_dynamic", action="store_true", help="comiple pytorch")
parser.add_argument("-compile_mode", default="default")
parser.add_argument("-torch_dtype", default="bfloat16")
parser.add_argument("-thr", type=float, default=0.5)

# transformers
parser.add_argument("-evaluation_strategy", default='steps')
parser.add_argument("-save_strategy", default='steps')
parser.add_argument("-eval_steps", type=int, default=1000)
parser.add_argument("-eval_delay", type=int, default=0)

parser.add_argument("-bs", "--batch_size", type=int)
parser.add_argument("-mbs", "--min_batch_size", type=int)
parser.add_argument("-vbs", "--val_batch_size", type=int)
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-lr_scheduler", default='linear')
parser.add_argument("-lr_scheduler_paras", type=json.loads, help='lr scheduler parameters')
parser.add_argument("-lr_warmup_ratio", type=float, default=0.0)
parser.add_argument("-warmup_steps", type=int, default=0)
parser.add_argument("-lr_decay_rate", type=float, default=1.0)

parser.add_argument("-init_kl_coef", type=float, default=0.2)
parser.add_argument("-max_grad_norm", type=float, default=1)
parser.add_argument("-gas", "--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("-opt", default="torch.optim.AdamW")
parser.add_argument("-optim", default="adamw_torch")
parser.add_argument("-optim_target_modules", nargs="+", default=None)
parser.add_argument("-opt_paras", type=json.loads, default={}, help='parameters for optimizer')
parser.add_argument("-n_dl_worker", type=int, default=0)
parser.add_argument("-n_task", type=int, default=4)
parser.add_argument("-weight_decay", type=float, default=1e-2)
parser.add_argument("-output_dir", default="../data")
parser.add_argument("-output_name")
parser.add_argument("-norm_func")
parser.add_argument("-verbose", type=int, default=16)
parser.add_argument("-report_to", default="tensorboard")
parser.add_argument("-min_cnt", type=int, default=0)
parser.add_argument("-output_keys", nargs='+')
parser.add_argument("-val_by_score",  action="store_true")
parser.add_argument("-val_steps", type=int, default=1000)
parser.add_argument("-init_epoch", type=int, default=0)
parser.add_argument("-epochs", type=int, default=10)
parser.add_argument("-n_init_epoch", type=int, default=0)
parser.add_argument("-n_epoch_step", type=int, default=10000000000)
parser.add_argument("-n_val_epoch_step", type=int, default=10000000000)
parser.add_argument("-es", "--early_stopping_patience", type=int, default=1)
parser.add_argument("-es_min_delta", type=float, default=0.0005)
parser.add_argument("-num", type=int, default=10000000000)
parser.add_argument("-train_num", type=int, default=10000000000)
parser.add_argument("-val_num", type=int, default=10000000000)
parser.add_argument("-ema", type=float, default=0)
parser.add_argument("-ema_start", type=float, default=0)
parser.add_argument("-topp", type=float, default=1.0)
parser.add_argument("-topk", type=int, default=-1)
parser.add_argument("-n_best", type=int, default=1)
parser.add_argument("-max_temp", type=float, default=5)
parser.add_argument("-min_temp", type=float, default=0.01)
parser.add_argument("-temp_decay_step", type=int, default=10000)
parser.add_argument("-n_frozen", type=int, default=0)
parser.add_argument("-frozen_emb",  action="store_true")
parser.add_argument("-use_lora",  action="store_true")
parser.add_argument("-lora_rank", type=int, default=32)
parser.add_argument("-lora_alpha", type=int, default=16)
parser.add_argument("-lora_dropout", type=float, default=0)
parser.add_argument("-lora_modules", nargs='+', default=None)
parser.add_argument("-lora_init")
parser.add_argument("-use_dora",  action="store_true")
parser.add_argument("-freeze_lm",  action="store_true")
parser.add_argument("-unfreeze_lm_head",  action="store_true")
parser.add_argument("-use_rslora",  action="store_true",default=None)
parser.add_argument("-ext_ratio",  type=float, default=0)
parser.add_argument("-use_orcamath",  action="store_true")
parser.add_argument("-use_mustard",  action="store_true")
parser.add_argument("-restore",  action="store_true")
parser.add_argument("-restore_model",  default="NA")
parser.add_argument("-restore_step",  type=int)
parser.add_argument("-do_train",  action="store_true")
parser.add_argument("-do_val",  action="store_true")
parser.add_argument("-eval_strategy")
parser.add_argument("-do_test",  action="store_true")
parser.add_argument("-do_concat",  action="store_true", default=None)
parser.add_argument("-nt", "--no_train",  action="store_true")
parser.add_argument("-to_list",  action="store_true", default=None)
parser.add_argument("-predict_val",  action="store_true")
parser.add_argument("-scoring",  action="store_true")
parser.add_argument("-predict_test",  action="store_true")
parser.add_argument("-use_unsloth",  action="store_true")
parser.add_argument("-use_dora",  action="store_true")
parser.add_argument("-use_4bit", action="store_true")
parser.add_argument("-use_8bit", action="store_true")
parser.add_argument("-hard_ratio", type=float, default=0)
parser.add_argument("-suffix", default="")
parser.add_argument("-split", type=int, default=-1)
parser.add_argument("-split_id", type=int, default=-1)
parser.add_argument("-n_split", type=int, default=2)
parser.add_argument("-mixup", type=float, default=0)
parser.add_argument("-use_semi", action="store_true")
parser.add_argument("-gpu_ratio", type=float, default=0.95)
parser.add_argument("-n_gpu", type=int, default=1)
parser.add_argument("-lora_fpath")
parser.add_argument("-n_vote", type=int)
parser.add_argument("-max_num_seqs", type=int, default=256)
parser.add_argument("-enforce_eager", action="store_true")
parser.add_argument("-save_lm_head", action="store_true")
parser.add_argument("-use_vllm", action="store_true")
parser.add_argument("-think_mode")
parser.add_argument("-url")


def contains_chinese(text: str) -> bool:
    """Return True if string contains at least one Chinese character."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def normalize_text(s: str) -> str:
    s = s.replace("‘", "'")
    s = s.replace("’", "'")
    tokens = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'")
    #s_list = [x.upper() if x in tokens else " " for x in s]
    s_list = [x.lower() if x in tokens else " " for x in s]
    s = " ".join("".join(s_list).split()).strip()
    return s

def get_text_id_hash(text):
    # Encode the string to bytes, then calculate the SHA256 hash
    hash_object = hashlib.sha256(text.encode('utf-8'))
    # Get the hash in hexadecimal format (a 64-character string)
    hex_digest = hash_object.hexdigest()
    return hex_digest


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        if self.training and 'mixup_w' in kwargs:
            if self.training and (self.mask_feature_prob > 0 or self.mask_time_prob > 0):
                input_features = self._mask_input_features(input_features, attention_mask=feature_attention_mask)
            mixup_input_features = kwargs.pop('mixup_input_features')
            mixup_w = kwargs.pop('mixup_w')
            mixup_input_ids  = kwargs.pop('mixup_input_ids')
            mixup_labels = kwargs.pop('mixup_labels')
            mixup_attention_mask = kwargs.pop('mixup_attention_mask')
            mixup_feature_attention_mask = kwargs.pop('mixup_feature_attention_mask')
            if self.training and (self.mask_feature_prob > 0 or self.mask_time_prob > 0):
                mixup_input_features = self._mask_input_features(mixup_input_features, attention_mask=mixup_feature_attention_mask)
            outputs = self.thinker.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                labels=labels,
                loss_kwargs=dict(reduction='none'),
                **kwargs,
            )
            mixup_outputs = self.thinker.forward(
                input_ids=mixup_input_ids,
                attention_mask=mixup_attention_mask,
                input_features=mixup_input_features,
                feature_attention_mask=mixup_feature_attention_mask,
                labels=mixup_labels,
                loss_kwargs=dict(reduction='none'),
                **kwargs,
            )
            outputs['loss'] = torch.mean(torch.sum(outputs['loss'], axis=1) * mixup_w / torch.sum(labels!=-100, axis=1) + (1-mixup_w)*torch.sum(mixup_outputs['loss'], axis=1)/torch.sum(mixup_labels!=-100, axis=1))
            return outputs


        else:
            if self.training and (self.mask_feature_prob > 0 or self.mask_time_prob > 0):
                input_features = self._mask_input_features(input_features, attention_mask=feature_attention_mask)
            return self.thinker.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                labels=labels,
                **kwargs,
            )

    cls.forward = forward
    cls._forward_patched = True

def score(preds, golds, is_word=True, normalize=True):
    num = len(preds)
    if is_word:
        preds = preds.merge(golds[["utterance_id", "orthographic_text"]])
        assert len(preds) == num
        s = score_wer(preds.orthographic_text, preds.pred, normalize=normalize)
    else:
        preds = preds.merge(golds[["utterance_id", "phonetic_text"]])
        assert len(preds) == num
        s = score_ipa_cer(preds.phonetic_text, preds.pred)
    return s



def load_data(args):
    if args.dataset=='csrw':
        if args.data_type=='train':
            df = pd.DataFrame(load_json_lines('../data/csrw/train_word_transcripts.jsonl'))
            df['audio_path'] = df.audio_path.apply(lambda x: f"../data/csrw/{x}")

            if args.use_tb:
                df2 = pd.DataFrame(load_json_lines('../data/csrp/train_word_transcripts.jsonl'))
                df2['audio_path'] = df2.audio_path.apply(lambda x: f"../data/csrp/{x}")
                df2 = df2[df2.utterance_id != 'U_b8a4e8220e65219b']
                df = pd.concat([df, df2])
            if not args.val_smoke:
                df = df[df.audio_duration_sec<(args.max_sec or 1000000)]
        elif args.data_type=='test':
            df = pd.DataFrame(load_json_lines('../data/csrw/utterance_metadata.jsonl'))
            df['audio_path'] = df.audio_path.apply(lambda x: f"../data/csrw/{x}")
        else:
            raise NotImplementedError(args.data_type)
    elif args.dataset == 'csrw2':
        df = pd.DataFrame(load_json_lines('../data/csrp/train_word_transcripts.jsonl'))
        df['audio_path'] = df.audio_path.apply(lambda x: f"../data/csrp/{x}")
        df = df[df.utterance_id != 'U_b8a4e8220e65219b']
        if not args.val_smoke:
            df = df[df.audio_duration_sec < (args.max_sec or 1000000)]
    elif args.dataset == 'csrw3':
        args2 = deepcopy(args)
        args2.dataset = 'csrw'
        args2.data_type = 'train'
        args2.use_tb = True
        args2.max_sec = None
        args2.num = 100000000
        csrw_a = load_data(args2)
        args2.max_sec = args.max_sec
        csrw_b = load_data(args2)
        df = csrw_a[~csrw_a.utterance_id.isin(csrw_b.utterance_id)]
        logger.info('csrw3: %s, %s, %s', len(csrw_a), len(csrw_b), len(df))

    elif args.dataset == 'csrp':
        if args.data_type=='train':
            df = pd.DataFrame(load_json_lines('../data/csrp/train_phon_transcripts.jsonl'))
            df['audio_path'] = df.audio_path.apply(lambda x: f"../data/csrp/{x}")
            if not args.val_smoke:
                df = df[df.audio_duration_sec < (args.max_sec or 1000000)]
            df = df[df.utterance_id!='U_b8a4e8220e65219b']
        elif args.data_type == 'test':
            df = pd.DataFrame(load_json_lines('../data/csrw/utterance_metadata.jsonl'))
            df['audio_path'] = df.audio_path.apply(lambda x: f"../data/csrw/{x}")
    elif args.dataset == 'csrp2':
        args2 = deepcopy(args)
        args2.dataset = 'csrw'
        args2.data_type = 'train'
        args2.use_tb = True
        args2.max_sec = None
        args2.num = 100000000
        csrw = load_data(args2)
        logger.info('all data %s', len(csrw))
        args2.dataset = 'csrp'
        args2.max_sec = args.max_sec
        csrp = load_data(args2)
        df = csrw[~csrw.utterance_id.isin(csrp.utterance_id)]
        logger.info('csrp2 has %s', len(df))
    elif args.dataset == 'trans':
        args2 = deepcopy(args)
        args2.dataset = 'csrp'
        csrp = load_data(args2)
        args2.dataset = 'csrw'
        args2.use_tb = True
        csrw = load_data(args2)
        csrw = csrw[csrw.utterance_id.isin(csrp.utterance_id)]
        num = len(csrp)
        df = csrp.merge(csrw[['utterance_id', 'orthographic_text']], on='utterance_id')
        assert len(df)==num
    elif args.dataset == 'pred':
        df = load_dump(args.pred_fpath)
        df['orthographic_text'] = df.pred.values
        if 'duration' not in df.columns:
            df['audio_duration_sec'] = -1
        else:
            df['audio_duration_sec'] = df.duration.values
        if args.data_type!='test' and 'phonetic_text' not in df.columns:
            args2 = deepcopy(args)
            args2.dataset = 'csrp'
            csrp = load_data(args2)
            num = len(df)
            df = df.merge(csrp[["utterance_id", "phonetic_text"]], on='utterance_id')
            assert num==len(df)


    elif args.dataset.startswith('ipap'):
        df = pd.read_csv('../data/ipap.csv')
        if args.dataset=='ipape':
            df = df[df.language=='en']
        df = df[df.audio_duration_sec < (args.max_sec or 1000000)]
        df.rename(columns={'clean': "phonetic_text"}, inplace=True)
        df['utterance_id'] = df['id']
        df['audio_path'] = df.apply(lambda x: f"../data/ipap/{x.ds}/{x.id}.flac" if os.path.exists(f"../data/ipap/{x.ds}/{x.id}.flac") else f"../data/ipap2/{x.ds}/{x.id}.flac", axis=1)
        df = df[df.audio_path.apply(lambda x: os.path.exists(x))]
    elif args.dataset.startswith('libriheavy'):
        if 'small' in args.dataset:
            df = pd.read_csv('../data/libriheavy_small.csv')
        elif 'medium' in args.dataset:
            df = pd.read_csv('../data/libriheavy_medium.csv')
        df = df[df.audio_duration_sec < (args.max_sec or 1000000)]
    else:
        raise NotImplementedError(args.dataset)
    df['src'] = args.dataset
    df['ID'] = range(len(df))
    df = df[:args.num]
    logger.info('num of %s is %s', args.dataset, len(df))
    return df


def restore_args(args, output_dir):
    restore_args = load_json(f"{output_dir}/args.json")
    new_args = []
    for k, v in vars(args).items():
        if v is None and k in restore_args:
            v = restore_args.get(k)
            setattr(args, k, v)
            new_args.append((k, v))
    logger.info("restored args:%s", new_args)



def get_vllm(args, modelid=None):
    kwargs = dict()
    if args.is_classify:
        kwargs['task'] = "classify"
    import vllm
    from vllm.lora.request import LoRARequest
    if modelid is None:
        modelid = args.model_name
    enable_lora, lora_request = False, None
    if args.lora_fpath is not None:
        lora_request = LoRARequest("lora", 1, args.lora_fpath)
        enable_lora = True

    elif os.path.exists(f"{modelid}/adapter_config.json"):
        from transformers import AutoConfig
        from peft import PeftConfig
        config = PeftConfig.from_pretrained(modelid)
        lora_request = LoRARequest("lora", 1, modelid)
        logger.info('use lora:%s, modelid:%s', modelid, config.base_model_name_or_path)
        modelid = config.base_model_name_or_path
        enable_lora = True

    llm = vllm.LLM(
        modelid,
        quantization=args.quantization,
        tensor_parallel_size=args.n_gpu,
        gpu_memory_utilization=args.gpu_ratio,
        trust_remote_code=True,
        dtype=args.torch_dtype,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_seq_len+args.max_gen_len+512,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=enable_lora,
        max_lora_rank=64,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
        **kwargs
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer, lora_request


def get_modelid(model_name):
    if 'KF' in model_name:
        modelid = sorted(glob(f"{model_name}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))[-1]
    else:
        modelid = model_name
    return modelid


def get_ckpt(output_dir, restore_step=None):
    if restore_step is not None:
        ckpt_dir = f"{output_dir}/checkpoint-{args.restore_step}"
    else:
        ckpt_dir = sorted(glob(f"{output_dir}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))[-1]
    return ckpt_dir


def set_logger(level=logging.INFO):
    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.setLevel(level)


def get_md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

@contextmanager
def timer(name):
    t0 = time.time()
    #print('{} start'.format(name))
    logger.info('%s start', name)
    yield
    #print('{} done in {} seconds'.format(name, time.time() - t0))
    logger.info('%s done in %s seconds', name, time.time()-t0)

def load_dump(fpath):
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    return data


def dump(data, fpath, protocol=2):
    fdir = os.path.dirname(fpath)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)


def load_json(fpath):
    with open(fpath) as f:
        dictionary = json.load(f)
    return dictionary


def load_json_lines(fpath, num=1e16, exclude_keys=None, post_process=None):
    if exclude_keys is None:
        exclude_keys = []
    data = []
    with open(fpath) as f:
        for l in f:
            dic = json.loads(l)
            for k in exclude_keys:
                if k in dic:
                    _ = dic.pop(k)
            if post_process is not None:
                dic = post_process(dic)
            data.append(dic)
            if len(data)>=num:
                break
    return data


def dump_json(dictionary, fpath, ensure_ascii=False):
    with open(fpath, 'w') as f:
        json.dump(dictionary, f, ensure_ascii=ensure_ascii)


def dump_json_lines(dicts, fpath, ensure_ascii=False):
    with open(fpath, 'w', encoding='utf8') as f:
        for d in dicts:
            json.dump(d, f, ensure_ascii=ensure_ascii)
            f.write(os.linesep)

def dynamic_import(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def shuffle_items(*args):
    inds = np.arange(len(args[0]))
    np.random.shuffle(inds)
    return [[x[ind] for ind in inds] for x in args]

def sorted_items(*args):
    return zip(*sorted(zip(*args), key=lambda x: x[0]))

def timestamp():
    return time.strftime('%Y%m%d%H%M%S')


def get_num_of_paras(m):
    num1, num2 = 0, 0
    for p in m.parameters():
        if p.requires_grad:
            num1 += p.numel()
        else:
            num2 += p.numel()
    return num1/1000/1000, num2/1000/1000


def load_audio(fpath, sr=None, mono=True, offset=0, duration=None):
    audio, sr = librosa.load(fpath, sr=sr, mono=mono, offset=offset, duration=duration)
    return audio, sr


def load_img(fpath, flag=None):
    if flag is None:
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(fpath, flag)
    return img




if __name__ == "__main__":
    args = parser.parse_args([])
    args.data_type = 'train'
    #df = load_data(args)
