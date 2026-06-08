from abc import ABC, abstractmethod
from pathlib import Path

from src.data.utils import load_audio, load_jsonl, save_jsonl
from src.paths import TARGET_SR


class BaseASRModel(ABC):

    @abstractmethod
    def inference(self, audio, sr=TARGET_SR):
        pass

    @abstractmethod
    def inference_batch(self, audio_list, sr=TARGET_SR):
        pass

    def inference_file(self, audio_path, sr=TARGET_SR):
        audio, sr = load_audio(audio_path, sr=sr)
        return self.inference(audio, sr)

    def inference_files(self, audio_paths, sr=TARGET_SR):
        audios = [load_audio(p, sr=sr)[0] for p in audio_paths]
        return self.inference_batch(audios, sr)

    def load_finetuned_weights(self, weights_path):
        raise NotImplementedError(f"{type(self).__name__} does not support loading finetuned weights")

    def load_checkpoint_weights(self, checkpoint_path):
        raise NotImplementedError(f"{type(self).__name__} does not support loading checkpoint weights")

    def predict_dataset(self, manifest_path, audio_dir):
        manifest = load_jsonl(manifest_path)
        audio_paths = [Path(audio_dir) / f"{e['utterance_id']}.flac" for e in manifest]
        texts = self.inference_files(audio_paths)
        return [
            {"utterance_id": e["utterance_id"], "orthographic_text": t}
            for e, t in zip(manifest, texts)
        ]

    def generate_submission(self, manifest_path, audio_dir, output_path):
        results = self.predict_dataset(manifest_path, audio_dir)
        save_jsonl(results, output_path)
