import torch 
import torch.nn as nn

class SpeakerEncoder(nn.Module):
    def __init__(self, mel_bins = 80, hidden = 256):
        super(SpeakerEncoder, self).__init__()

        self.lstm = nn.LSTM(input_size=mel_bins, hidden_size=hidden,
                            num_layers=3, batch_first=True )
        
        self.linear = nn.Linear(hidden, 256)
        self.relu = nn.ReLU()

    def forward(self , x):
        # X Shape : (Batch , time , mel)
        output, (hidden,_) = self.lstm(x)

        embedding = self.linear(hidden[-1])
        embedding = self.relu(embedding)

        return embedding