import tempfile
from pathlib import Path

import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel

from src.models.base import BaseASRModel
from src.paths import TARGET_SR
from tqdm import tqdm

class QwenASR(BaseASRModel):

    def __init__(self, model_name="Qwen/Qwen3-ASR-1.7B", device="cuda:0", dtype=None):
        self.dtype = dtype or torch.bfloat16
        self.qwen_model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=self.dtype,
            device_map=device,
            max_new_tokens=2048,
        )
        self.qwen_model.model.thinker.generation_config = self.qwen_model.model.generation_config

    def inference_file(self, audio_path, sr=TARGET_SR):
        results = self.qwen_model.transcribe(audio=str(audio_path), language="English")
        return results[0].text.strip()

    def inference(self, audio, sr=TARGET_SR):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, audio, sr)
            return self.inference_file(f.name, sr)

    def inference_batch(self, audio_list, sr=TARGET_SR):
        results = []
        for audio_file in tqdm(audio_list):
            result = self.qwen_model.transcribe(audio=audio_file, language="English")
            results.append(result.text.strip())
        return results
        # results = self.qwen_model.transcribe(audio=audio_list, language="English")
        # return [r.text.strip() for r in results]
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     paths = []
        #     for i, audio in enumerate(audio_list):
        #         p = Path(tmp_dir) / f"{i}.wav"
        #         sf.write(str(p), audio, sr)
        #         paths.append(str(p))
        #     return [self.inference_file(p, sr) for p in paths]
