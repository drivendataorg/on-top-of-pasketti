# Solution - On Top of Pasketti: Children's Speech Recognition Challenge Runtime - Phonetic Track
Username: dzunglt24

## Summary

**TL;DR**: I train a multi-task, hybrid CTC-based ASR model that jointly optimizes standard CTC loss with a 4-way consistency objective, improving both transcription accuracy and robustness to noise.

### Model Architecture  
Our architecture combines **W2v-BERT 2.0** and **WavLM-Large** to leverage complementary acoustic representations. I further apply a backbone dropout strategy during training to enhance generalization.

### Data  
I use only competition-provided data, including both **Word** and **Phonetic** tracks from DrivenData and TalkBank, along with the RealClass noise dataset.  

- Word track data is used for multi-task training with two CTC heads:  
  - one for word-level transcription  
  - one for IPA character prediction  
- For audio samples present in the Word track but missing in the Phonetic track, I generate pseudo IPA labels to augment training (Only keep pseudo label with score >= 0.7)


### Data Augmentation  
I apply audio augmentation using the *audiomentations* library, including pitch shifting, gain adjustment, and background noise injection:

```python
augmenter = Compose([
    PitchShift(min_semitones=-5.0, max_semitones=5.0, p=1),
    Gain(min_gain_db=-6, max_gain_db=6, p=1),
    AddBackgroundNoise(
        sounds_path=[str(NOISE_DIR)],
        min_snr_db=MIN_SNR_DB,
        max_snr_db=MAX_SNR_DB,
        p=1
    )
])
```

### Training Objective

The model is trained with a combination of standard CTC loss and a **4-way consistency loss** inspired by CR-CTC (CR_LOSS_SCALE=0.2).

For each input sample `A`, I generate an augmented version `B`. Both are passed through the model twice with different SpecAugment masks and dropout, producing `A'` and `B'`.

The consistency objective is defined as the sum of KL divergence across four pairs:

- (A, A')  
- (B, B')  
- (A, B)  
- (A', B')  

This encourages stability across both augmentation and stochastic model variations.

### Final submission: 
- Ensemble of 4 models: 1 trained with all labeled data and 3 trained with all+pseudo data 
    - Model 1: trained with all labeled data, ALPHA=0.75, MASK_TIME_PROB=0.15, MASK_FEATURE_PROB=0.05, MIN_SNR_DB=2, MAX_SNR_DB=10
    - Model 2: trained with all+pseudo data, ALPHA=0.8, MASK_TIME_PROB=0.3, MASK_FEATURE_PROB=0.05, MIN_SNR_DB=2, MAX_SNR_DB=10
    - Model 3: trained with all+pseudo data, ALPHA=0.75, MASK_TIME_PROB=0.3, MASK_FEATURE_PROB=0.05, MIN_SNR_DB=3, MAX_SNR_DB=10
    - Model 4: trained with all+pseudo data, ALPHA=0.75, MASK_TIME_PROB=0.15, MASK_FEATURE_PROB=0.05, MIN_SNR_DB=2, MAX_SNR_DB=10
    - All model trained with EPOCHS = 5, LR = 2e-5, CR_LOSS_SCALE=0.2, MAX_DURATION=30, BATCH_SIZE=2, ACCUMULATE=4 
- Public: 0.2618 Private: 0.2629


### Improvements on Public Leaderboard: Road from 0.2846 to 0.2618
- **Add 4-way consistency loss**: ~0.015 CER reduction
- **From W2v-BERT to Hybrid W2v-BERT+WavLM**: ~0.002 CER reduction
- **Multi-task training**: ~0.0002 CER reduction  
- **Pseudo-labeling**: ~0.002 CER reduction  
- **Ensembling (3–4 models, best-score selection)**: ~0.003–0.004 CER reduction 


# Setup

1. Install the required package

- ```pip install -r requirement.txt```

2. Prepare the data
- Copy the competition data into: 
    - data/audio_drivendata
    - data/audio_drivendata_phon
    - data/audio_talkbank
    - data/train_phon_transcripts_drivendata.jsonl
    - data/train_phon_transcripts_talkbank.jsonl
    - data/train_word_transcripts_drivendata.jsonl
    - data/train_word_transcripts_talkbank.jsonl

- Resample noise dataset to 16k sampling rate: 
    - ```python resampling_noise.py```

- Format of training data file:  {"audio_filepath": "...", "duration": ..., "text": "ʔə ʔæpɫ", "word": "a apple"}

    - Run ```python process_train_data.py```
        - Training data: all audios with word-level and IPA transcriptions (phon_train_all_filtered.json)
        - Data for pseudo label: all audios with word-level but no IPA transcriptions (nolabel_phon.json)


# Hardware:
The solution was run on 2 H200s (can run on H100 with BATCH_SIZE=1)

Training time: ~33 hours for 5 epochs with pseudo labels. Without pseudo labels, ~10 hours for 5 epochs.

Inference time: 25 minutes submission time for single hybrid model. Ensemble 4 models take 1 hour 40 minutes.

# Run training
1. Training
- ```python train_multi_4way_hybrid.py```

2. Pseudo label: 
- Generate from audios with word-level transcription but no IPA transcription.
    - ```python pseudo_label.py```
        - Pseudo label file: pseudolabel_phon.json
- Merge train data with pseudo label data
    - ```python merge_pseudolabel.py```
        - Final pseudo+train file: phon_train_all_filtered_pseudo07.json

# Run inference
1. Trained Model weights: Download and place it to submission/models folder
- Model 1: https://drive.google.com/file/d/1D13ffXZv8dERctenvrgS8oOXlE5joB4N/view?usp=sharing
- Model 2: https://drive.google.com/file/d/1_HKNYn1xIkxVxXkHqPisOgVgyQOZa2li/view?usp=sharing
- Model 3: https://drive.google.com/file/d/1W5PKyOH9H-pCrVq7kwTPLDSVgBO6uhBs/view?usp=sharing
- Model 4: https://drive.google.com/file/d/1QQxn8ZsIqEklxDUaMgv6qnblFquuMBgh/view?usp=sharing

2. Inference code
- See submission folder
- Run ```python main.py```