# Solution - [On Top of Pasketti: Children’s Speech Recognition Challenge - Word Track]

Username: chuxiliyixiaosa
Rank: 3

## Summary

- Use competition data and TalkBank corpus data to fine-tune Qwen/Qwen3-ASR-1.7B.
- vLLM inference takes about 1 hour to finish.

# Setup
1.conda create -n asr python=3.10
2.conda activate asr
3.pip install -r requirements.txt

# Hardware

The solution was run on 4 NVIDIA GeForce RTX 4090 D

Training time: 60 hour

Inference time: 1 hour

# Run training

1.Download audio_part_0.zip, audio_part_1.zip, audio_part_2.zip, and train_word_transcripts.jsonl from https://www.drivendata.org/competitions/308/childrens-word-asr/data/ to the data/ directory, and unzip the .zip files.
2.Download audio.zip and train_word_transcripts.jsonl from https://media.talkbank.org/childes/0extra/DrivenData/ to the data/TalkBank_corpus/ directory, and unzip the .zip files.
3.Download https://huggingface.co/Qwen/Qwen3-ASR-1.7B into the ./ directory.
4.Run 'train-X4090D-v013-Qwen3-ASR-1.7B-all-data.ipynb'
    - Once training reaches 85,950 steps and the checkpoint is saved, you can stop the training.
5.Run 'train-X4090D-v025-X4090D-v013-continue-training.ipynb'

- How much space will your model weights file(s) require? 
answer: 40 GB
- Where will model weights be saved out to by default? 
answer: data/models/
- How can we access your trained model weights? We only rerun the inference step, so we should be able to download any model files needed without rerunning the full training process. 
answer: https://huggingface.co/chuxiliyixiaosa/children_asr_word_track_rank3

# Run inference

1.Download submission_X4090D-v025.zip from https://huggingface.co/chuxiliyixiaosa/children_asr_word_track_rank3 into the ./ directory, and unzip the .zip files.
2.```shell
./vllm/bin/python3.11 ./infer.py
```
- It will read the input file /code_execution/data/utterance_metadata.jsonl and save the results to /code_execution/submission/submission.jsonl
- The required inference environment has already been set up in the ./vllm directory.
- qwen_asr: I have modified the code to read all audio files into memory in parallel for inference.

# Other explorations
- Since I only need 1 hour to complete inference on 220,000 audio samples, while the competition's maximum time limit is 2 hours, I later kept trying larger models (Qwen2.5-omni-7B, Kimi-Audio-7B) to infer on 5% of the audio, but never gained any score improvement.