import os, sys, logging
import re
import torch
from glob import glob
from tqdm import tqdm
import inspect
from itertools import combinations
import numpy as np
import pandas as pd
import util
from multiprocessing import Pool
from lxml import html
from collections import defaultdict
import xml.etree.ElementTree as ET
import requests
import httpx
from functools import partial
import time
import soundfile as sf
import hashlib
import random
import shutil
import torch.distributed as dist
import math
import torch

logger = logging.getLogger(__name__)
ppt = '''Your task is to mimic an {} years old {} speaking in classroom, please generate the speaking text as below examples. 
Please do not repeat the topic in the examples.
Example 1:
{}
Example 2:
{}
Example 3:
{}
Example 4:
{}
Example 5:
{}
Please generate the speaking text about a new topic, please directly output the text without any explanation.
'''


def set_seed(seed, benchmark=False, deterministic_algorithms=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    if benchmark:
        torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class IterDataset(torch.utils.data.IterableDataset):
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

    def __init__(self, args, df, output_dir):
        self.args = args
        self.data = df.to_records(index=False)
        self.output_dir = output_dir


    def getitem(self, index, rec=None):
        if rec is None:
            rec = self.data[index]
        text_id = util.get_text_id_hash(rec.text)
        fpath = f"{self.output_dir}/{text_id}.wav"
        age = np.random.randint(3, 16)
        sex = np.random.choice(['boy', 'girl'], replace=False)
        if os.path.exists(fpath):
            return None
        if self.args.split_id>=0 and (index%self.args.n_split)!=self.args.split_id:
            return None
        instruct = "Please mimic an {} years old {} speaking".format(age, sex)
        item = dict()
        item['instruct'] = instruct
        item['text'] = rec.text
        item['fpath'] = fpath
        return item

    def collate(self, batch):
        batch = {k:[item[k] for item in batch]for k in batch[0].keys()}
        return batch


class AugCloneDataset(IterDataset):
    def __init__(self, args, df, output_dir):
        super().__init__(args, df, output_dir)
        self.cids = sorted(df.child_id.unique())
        self.data = sorted(df.orthographic_text.unique())
        self.exists = set(zip(df.child_id, df.orthographic_text))
        self.refs = {cid: gp.to_records(index=False) for cid, gp in df.groupby('child_id')}

    def __len__(self):
        return len(self.data)*self.args.n_clone

    def get_iter_items(self, index):
        if self.args.split_id>=0 and (index%self.args.n_split)!=self.args.split_id:
            return None
        text = self.data[index]
        n = 0
        while n<args.n_clone:
            item = dict(text=text)
            cid = np.random.choice(self.cids)
            if not (cid, text) in self.exists:
                n += 1
                text_id = util.get_text_id_hash(text)
                fpath = f"{self.output_dir}/{text_id}_{cid}.wav"
                _refs = self.refs[cid]
                ref = np.random.choice(_refs)
                if not os.path.exists(fpath):
                    audio, sr = util.load_audio(ref.audio_path, sr=16000)
                    #audio = audio / (np.max(np.abs(audio)) + 1e-8)
                    item['ref_audio'] = (audio, sr)
                    item['ref_text'] = ref.orthographic_text
                    item['language'] = "English"
                    item['fpath'] = fpath
                    yield item



def _estimate_prompt_len(
    additional_information,
    model_name,
    _cache = {},
) -> int:
    """Estimate prompt_token_ids placeholder length for the Talker stage.

    The AR Talker replaces all input embeddings via ``preprocess``, so the
    placeholder values are irrelevant but the **length** must match the
    embeddings that ``preprocess`` will produce.
    """
    try:
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )

        if model_name not in _cache:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
            cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)
            _cache[model_name] = (tok, getattr(cfg, "talker_config", None))

        tok, tcfg = _cache[model_name]
        task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]
        return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
            additional_information=additional_information,
            task_type=task_type,
            tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
            codec_language_id=getattr(tcfg, "codec_language_id", None),
            spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
        )
    except Exception as exc:
        logger.warning("Failed to estimate prompt length, using fallback 2048: %s", exc)
        return 2048

def to_16k_noise(args):
    output_dir = "../data/preprocessed/noise_16k"
    for fpath in tqdm(glob(f"../data/csrw/noise/audio/*.flac"), desc="to 16k"):
        fname = os.path.basename(fpath)
        audio, _ = util.load_audio(fpath, sr=16000, mono=True)
        out_fpath = f"{output_dir}/{fname}"
        sf.write(out_fpath, audio, 16000, subtype='PCM_16')


def gen_aug_clone_vllm_omni(args):
    from vllm_omni import Omni
    output_dir = f"../data/preprocessed/clone_data"

    np.random.seed(args.seed)
    num = args.num
    args.num = 100000000
    df = util.load_data(args)
    args.num = num

    cids = sorted(df.child_id.unique())
    texts = sorted(df.orthographic_text.unique())
    exists = set(zip(df.child_id, df.orthographic_text))
    os.makedirs(output_dir, exist_ok=True)
    import torch
    import soundfile as sf

    refs = {cid: gp.to_records(index=False) for cid, gp in df.groupby('child_id')}
    inputs = []
    for text in tqdm(texts):
        np.random.shuffle(cids)
        n = 0
        for cid in cids:
            _refs = refs[cid]
            if not (cid, text) in exists:
                n += 1
                text_id = util.get_text_id_hash(text)
                fpath = f"{output_dir}/{text_id}_{cid}.wav"
                ref = np.random.choice(_refs)
                if os.path.exists(fpath):
                    if n >= 2:
                        break
                    continue

                additional_information = {
                    "task_type": ['Base'],
                    "ref_audio": [ref.audio_path],
                    "ref_text": [ref.orthographic_text],
                    "text": [text],
                    "language": ['English'],
                    "x_vector_only_mode": [False],
                    "max_new_tokens": [2048],
                }
                prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
                inputs.append({
                    "prompt": prompt,
                    "additional_information": additional_information,
                    "fpath": f"{output_dir}/{text_id}_{cid}.wav",
                    }
                )

                if n >= 2:
                    break
    print('prepare inputs done')
    omni = Omni(
        model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        stage_configs_path=None,
        log_stats=False,
        stage_init_timeout=300,
    )

    batch_size = args.batch_size
    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="batch gen"):
        batch = inputs[batch_start : batch_start + batch_size]
        fpaths = [item.pop('fpath') for item in batch]
        omni_generator = omni.generate(batch, sampling_params_list=None)
        for fpath, stage_outputs in zip(fpaths, omni_generator):
            num = 0
            for output in stage_outputs.request_output:
                num += 1
                request_id = output.request_id
                audio_data = output.outputs[0].multimodal_output["audio"]
                # async_chunk mode returns a list of chunks; concatenate them.
                if isinstance(audio_data, list):
                    audio_tensor = torch.cat(audio_data, dim=-1)
                else:
                    audio_tensor = audio_data
                sr_val = output.outputs[0].multimodal_output["sr"]
                audio_samplerate = sr_val.item() if hasattr(sr_val, "item") else int(sr_val[-1])
                # Convert to numpy array and ensure correct format
                audio_numpy = audio_tensor.float().detach().cpu().numpy()

                # Ensure audio is 1D (flatten if needed)
                if audio_numpy.ndim > 1:
                    audio_numpy = audio_numpy.flatten()

                # Save audio file with explicit WAV format
                sf.write(fpath, audio_numpy, samplerate=audio_samplerate, format="WAV")
            assert num<2


def gen_aug_clone(args):
    set_seed(args.seed)
    num = args.num
    args.num = 100000000
    df = util.load_data(args)
    args.num = num

    output_dir = f"../data/preprocessed/clone_data"
    os.makedirs(output_dir, exist_ok=True)

    ds = AugCloneDataset(args, df, output_dir)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                     shuffle=False, drop_last=False, collate_fn=ds.collate)

    assert args.batch_size == 1

    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        #torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    for batch in tqdm(dl, desc='gen aug clone'):
                wavs, sr = model.generate_voice_clone(
                    text=batch['text'][0],
                    language=batch['language'][0],
                    ref_audio=tuple(batch['ref_audio'][0]),
                    max_new_tokens=1024,
                    x_vector_only_mode=True,
                )
                sf.write(batch['fpath'][0], wavs[0], sr)
    print("Done!")





def gen_text(args):
    np.random.seed(args.seed)
    num = args.num
    args.num = 100000000
    df = util.load_data(args)
    args.num = num
    output_dir = os.path.join(args.data_dir, 'preprocessed', args.gen_dir)
    os.makedirs(output_dir, exist_ok=True)
    llm, tokenizer, lora_request = util.get_vllm(args, args.model_name)
    recs = df.to_records(index=False)
    inds = np.arange(len(recs))
    ppts = []
    n1 = len(inds)//5
    n2 = args.num//n1 + 1
    inds_ = []
    for i in range(n2):
        np.random.shuffle(inds)
        for j in range(0, len(inds), 5):
            inds_ += list(inds[j:j+5])
    print('inds generated', len(inds_))
    ages = np.random.randint(3, 16, args.num)
    sexs = np.random.choice(['boy', 'girl'], args.num, replace=True)
    from vllm import SamplingParams
    sp = SamplingParams(seed=args.seed, temperature=args.temp, skip_special_tokens=True, max_tokens=100, top_p=args.topp)
    for i in tqdm(range(args.num), desc='generate ppts'):
        ind1, ind2, ind3, ind4, ind5 = inds_[i*5:(i+1)*5]
        rec1, rec2, rec3, rec4, rec5 = recs[ind1], recs[ind2], recs[ind3], recs[ind4], recs[ind5]
        age = ages[i]
        sex = sexs[i]
        ppt_ = ppt.format(age, sex, rec1.orthographic_text, rec2.orthographic_text, rec3.orthographic_text, rec4.orthographic_text, rec5.orthographic_text)
        msgs = [{"role": "user", "content": ppt_}]
        ppts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    print(ppts[:2])
    outputs = llm.generate(ppts, sp, use_tqdm=True, lora_request=lora_request)
    outputs = [output.outputs[0].text for output in outputs]
    logger.info('vllm outputs:%s', '-----------\n'.join(outputs[:2]))
    df = pd.DataFrame(outputs, columns=['text'])
    df['ppt'] = ppts
    fpath = f"{output_dir}/gen_text_{args.seed}.csv"
    df.to_csv(fpath, index=False)
    print(f'Done, {fpath}')


def gen_audio(args):
    set_seed(args.seed)
    output_dir = os.path.join(args.data_dir, 'preprocessed', 'gen_audio')
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    fpaths = sorted(glob(f"{args.data_dir}/preprocessed/{args.gen_dir}/gen_text*.csv"))
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    dfs = []
    for fpath in fpaths:
        df = pd.read_csv(fpath)
        dfs.append(df)
    df = pd.concat(dfs)
    df['text'] = df.text.str.lower()
    df = df[~df.text.apply(util.contains_chinese)]
    df = df.groupby('text').head(1)
    df['l'] = df.text.apply(lambda x: len(x.split()))
    df = df.sort_values('l', ascending=False)
    #df['text_id'] = df.text.apply(util.get_text_id_hash)
    #exists = set([os.path.basename(fpath).split('.wav')[0] for fpath in glob(f'{output_dir}/*.wav')])
    #df = df[~df.text_id.isin(exists)]
    ds = IterDataset(args, df, output_dir)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, pin_memory=True, num_workers=args.n_dl_worker,
                                     shuffle=False, drop_last=False, collate_fn=ds.collate)
    for batch in tqdm(dl, desc='gen audio'):
        wavs, sr = model.generate_voice_design(
            text=batch['text'],
            language= ["English"]*len(batch['text']),
            instruct=batch['instruct'],
        )
        for fpath, wav in zip(batch['fpath'], wavs):
            sf.write(fpath, wav, sr)



if __name__ == "__main__":
    args = util.parser.parse_args()
    gl = globals()
    if args.debug:
        util.set_logger(logging.DEBUG)
    else:
        util.set_logger()
    if args.method_name in gl:
        gl[args.method_name](args)
    else:
        logging.error('unknown method : %s', args.method_name)
