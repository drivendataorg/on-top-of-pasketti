# Summary
My solution is simply fine-tuning Qwen/Qwen3-ASR-1.7B with only competition data.

I use spec-aug and competition noise data to improve model generalize ablility.

I use audios less than 45 seconds. For other audios they were split by 45 seconds and predict the semi label.

My best submission was ensemble(model soup) of 18 models.



# Environment
Ubuntu 22.04, 1xRTX4090 24G, CPU 24 core 64G, python 3.12



# install
- unzip  word track data and noise to data/csrw. data/csrw/audio contain audios of the track, data/csrw/noise/audio contain the competition noise audios
  ```
  ls
  audio   train_word_transcripts.jsonl
  noise  

  ```
- unzip phonetic track data to data/csrp.
  ```
  ls
  audio     train_phon_transcripts.jsonl
  train_word_transcripts.jsonl

  ```
- copy flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl  to data/wheels
  
- run ./setup.sh install requirements
  
- all program output will save under folder data

# training
Training detail please refer csrw/train.ipynb. If you have not download the qwen3-asr model weight, please remove TRANSFORMERS_OFFLINE=1.

## preprocess noise to 16k

## stage1
- train 5 models with all competition data (~ 40 hours)
- create model soup CSRW_qw3asr_1d7b_d19_ms_KF0
- create semi (~ 1 hour)
  
## stage2
- train CSRW_qw3asr_1d7b_d24 with semi data (~ 40 hours)
- train CSRW_qw3asr_1d7b_d25 with semi data but disable noise (~ 64 hours)
  
## create model soup CSRW_qw3asr_1d7b_d25_ms_KF0

# generate submission.zip for inference(output to data/submission_CSRW). The best submission takes ~118 minutes on drivendata env(A100).
./gen_submit.sh CSRW_qw3asr_1d7b_d25_ms_KF0


  
