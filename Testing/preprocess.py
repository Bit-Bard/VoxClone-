import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_path = "C:\\Users\\dhruv\\OneDrive\\Documents\\Desktop\\Deep Learning Project\\VoxClone_3\Mini_Dataset\\LJ001-0001.wav"

# load audio
audio, sr = librosa.load(audio_path, sr=22050)

# create mel spectrogram
mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80
)

# convert to log scale
mel_db = librosa.power_to_db(mel, ref=np.max)

print("Mel shape:", mel_db.shape)

# visualize
plt.figure(figsize=(10,4))
plt.imshow(mel_db, aspect='auto', origin='lower')
plt.title("Mel Spectrogram")
plt.colorbar()
plt.show()