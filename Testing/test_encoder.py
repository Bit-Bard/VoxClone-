import torch
from speaker_encoder import SpeakerEncoder

# fake mel spectrogram
mel = torch.randn(1, 832, 80)

model = SpeakerEncoder()

embedding = model(mel)

print("Embedding shape:", embedding.shape)