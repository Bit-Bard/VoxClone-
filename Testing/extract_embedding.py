import torch
import librosa
import numpy as np
from speaker_encoder import SpeakerEncoder

audio_path = "C:\\Users\\dhruv\\OneDrive\\Documents\\Desktop\\Deep Learning Project\\VoxClone_3\Mini_Dataset\\LJ001-0001.wav"

# load audio
audio, sr = librosa.load(audio_path, sr=22050)

# mel spectrogram
mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80
)

mel_db = librosa.power_to_db(mel, ref=np.max)

# convert to tensor
mel_tensor = torch.tensor(mel_db).T.unsqueeze(0).float()

# load encoder
model = SpeakerEncoder()

# generate embedding
embedding = model(mel_tensor)

print("Embedding shape:", embedding.shape)
print("Embedding vector:", embedding)