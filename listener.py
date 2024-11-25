import sounddevice as sd
import numpy as np
import librosa
import threading
import os
from datetime import datetime
import matplotlib.pyplot as plt

SPECTROGRAM_DIR = 'NewSpec' 

# Parameters
sr = 44100  # Sampling rate
channels = 1  # Number of channels (mono)
dtype = 'float32'  # Data type for recording
is_recording = True  # Flag to control recording


def stop_recording():
    """Function to stop the recording when a key is pressed."""
    global is_recording
    input("Press Enter to stop recording...\n")
    is_recording = False


def record_audio():
    """Function to record audio until stopped."""
    global is_recording
    print("Recording... Press Enter to stop.")
    
    # Create a buffer for audio data
    audio_buffer = []

    # Callback function for audio input
    def callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        if is_recording:
            audio_buffer.append(indata.copy())
        else:
            raise sd.CallbackStop()

    # Start the recording stream
    with sd.InputStream(samplerate=sr, channels=channels, dtype=dtype, callback=callback):
        while is_recording:
            sd.sleep(100)  # Small delay to keep the loop responsive

    print("Recording stopped!")
    
    # Concatenate the buffer into a single array
    audio = np.concatenate(audio_buffer, axis=0).flatten()
    return audio

def save_spectrogram(audio, sr, output_path):
    spectrogram = librosa.stft(audio)
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.axis('off') 
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Spectrogram saved to: {output_path}")

def process_audio_file(audio, sr):
    output_filename = f"{datetime.now()}_spectrogram.png"
    output_path = os.path.join(SPECTROGRAM_DIR, output_filename)
    save_spectrogram(audio, sr, output_path)


# Start a thread to listen for the stop command
stop_thread = threading.Thread(target=stop_recording)
stop_thread.start()

# Record audio
audio = record_audio()

# Normalize the audio (optional)
audio = librosa.util.normalize(audio)

# Output the results
print(f"Audio recorded: {audio.shape}, Sampling rate: {sr}")

process_audio_file(audio, sr)