import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(input_size,hidden_size)

    def forward(self, input):
        out = self.fc(input)
        out, _ = self.lstm(out)
        out = nn.LSTM.dropout(0.2)
        out = self.fc(input)