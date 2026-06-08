[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[<img src='https://drivendata-prod-public.s3.amazonaws.com/comp_images/gates-asr-group_copy.jpg' width='600'>](https://www.drivendata.org/competitions/group/childrens-asr-competition/)

# On Top of Pasketti: Children's Speech Recognition Challenge

## Goal of the Competition

Automatic speech recognition (ASR) models transcribe adult speech well but struggle with children's voices. Kids have distinct vocal characteristics, inconsistent pronunciation, and are still developing the motor skills that shape how they speak — resulting in error rates 4–8x worse than for adults. This performance gap has left educators without reliable tools for the early education applications that stand to benefit most from automated speech analysis.

**The On Top of Pasketti: Children's Speech Recognition Challenge** brought together a global community of machine learning practitioners to develop open ASR models tailored to early education. 

The challenge ran two tracks: the [**Word Track**](https://www.drivendata.org/competitions/308/childrens-word-asr/) focused on accurate word-level transcription to enable automated transcription, verbal tool use, and assessments of comprehension and reasoning; and the [**Phonetic Track**](https://www.drivendata.org/competitions/309/childrens-phonetic-asr/) focused on capturing the sounds children actually produce, which is critical for diagnostic applications like speech pathology screening.

## What's in this Repository

This repository contains code from winning competitors in the [On Top of Pasketti: Children's Speech Recognition Challenge](https://www.drivendata.org/competitions/group/childrens-asr-competition/) on DrivenData. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

### Word Track

Place | Team or User | WER | Noisy WER | Summary of Model
--- | --- | --- | --- | ---
1st and noisy bonus | [ktrw](https://www.drivendata.org/users/ktrw/) | 0.1937 | 0.4950 | Used WER from a fine-tuned Parakeet model to generate quality-stratified datasets, then ensembled LoRA fine-tuned Qwen-3-ASR-1.7b models trained on each.
2nd and noisy bonus | [legend](https://www.drivendata.org/users/legend/) | 0.1953 | 0.4791 | Fine-tuned 18 Qwen3-ASR-1.7B using LoRA and ensembled models using weight averaging ("model soup").
3rd | [chuxiliyixiaosa](https://www.drivendata.org/users/chuxiliyixiaosa/) | 0.1984 | 0.4970 | Fine-tuned Qwen3-ASR-1.7B.
Noisy bonus | [mitchelld12345](https://www.drivendata.org/users/mitchelld12345/) | 0.2248 | 0.4868 | Fine-tuned Qwen3-ASR-1.7B with KL distillation and TTS-generated synthetic single-word training data.
Noisy bonus | [shiqi_47](https://www.drivendata.org/users/shiqi_47/) | 0.2115 | 0.4919 | Full-parameter fine-tuned Qwen3-ASR-1.7B on competition data plus TalkBank children's speech corpus.

### Phonetic Track

Place | Team or User | IPA-CER | Summary of Model
--- | --- | --- | ---
1st | [gezi](https://www.drivendata.org/users/gezi/) | 0.2559 | Trained 11 NeMo Parakeet and WavLM models using a dual-head to jointly learn IPA and word-level outputs and ensembled with a CatBoost reranker.
2nd | Team Epoch VI: [reinmv](https://www.drivendata.org/users/reinmv/), [Max28](https://www.drivendata.org/users/Max28/), [WillemDieleman](https://www.drivendata.org/users/WillemDieleman/) | 0.260728 | Trained 13 WavLM, HuBERT, and Whisper models with a 2-layer CTC head and ensembled using ROVER.
3rd | [dzunglt24](https://www.drivendata.org/users/dzunglt24/) | 0.2629 | Trained 4 W2v-BERT and WavLM models using CTC loss with a four-way consistency objective.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Winners Blog Post: [Meet the winners of the On Top of Pasketti: Children's Speech Recognition Challenge](https://drivendata.co/blog/on-top-of-pasketti-winners)**

**Reference Implementations: [Word Track](https://drivendata.co/blog/child-asr-word-benchmark) | [Phonetic Track](https://drivendata.co/blog/child-asr-phonetic-reference-implementation)**
