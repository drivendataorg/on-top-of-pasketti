import json

import librosa


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr, dtype="float32", mono=True)
    return audio, sr
