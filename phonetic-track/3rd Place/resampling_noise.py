import os
import glob
import librosa
import soundfile as sf
import numpy as np

def resample_audio_directory(input_dir, output_dir, target_sample_rate=16000):
    """
    Reads all audio files from an input directory, resamples them to a target sample rate,
    and saves them to an output directory.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Supported audio extensions (librosa supports many formats via ffmpeg/avconv)
    extensions = ('*.wav', '*.mp3', '*.flac', '*.aiff')
    files_to_process = []
    for ext in extensions:
        files_to_process.extend(glob.glob(os.path.join(input_dir, ext)))

    if not files_to_process:
        print(f"No audio files found in '{input_dir}' with supported extensions.")
        return

    print(f"Found {len(files_to_process)} files to process.")

    for filepath in files_to_process:
        try:
            # 1. Read audio file (librosa automatically resamples to 22.05 kHz by default, 
            #    so we must load with the original sample rate first (sr=None))
            # The 'mono=False' option preserves original channels if desired
            audio_data, original_sr = librosa.load(filepath, sr=None, mono=False)

            # Ensure the original sample rate is 48kHz for this specific task
            if original_sr != 48000:
                print(f"Skipping '{os.path.basename(filepath)}', unexpected sample rate: {original_sr} Hz")
                continue

            # 2. Resample the audio data to 16kHz
            # librosa.resample performs high-quality resampling with anti-aliasing filters
            if audio_data.ndim > 1:
                # Resample multi-channel audio
                resampled_audio = np.vstack([
                    librosa.resample(channel, orig_sr=original_sr, target_sr=target_sample_rate)
                    for channel in audio_data
                ])
            else:
                # Resample mono audio
                resampled_audio = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sample_rate)

            # 3. Define the output path and save the file (using soundfile for reliable saving)
            filename = os.path.basename(filepath)
            output_filepath = os.path.join(output_dir, filename)
            
            # soundfile automatically uses 16-bit PCM for WAV files by default, suitable for the target
            sf.write(output_filepath, resampled_audio.T if resampled_audio.ndim > 1 else resampled_audio, target_sample_rate)

            print(f"Successfully resampled and saved: {output_filepath}")

        except Exception as e:
            print(f"Error processing file '{filepath}': {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define your input and output directories
    source_directory = "data/noise"
    destination_directory = "data/noise_16k"
    
    # Run the function
    resample_audio_directory(source_directory, destination_directory, target_sample_rate=16000)
