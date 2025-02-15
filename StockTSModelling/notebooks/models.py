import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        

        # adjust fully connected layer output size for bi-LSTM
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)

    def forward(self, x):
        device = x.device
        num_directions = 2 if self.bidirectional else 1  # Bi-LSTM needs twice the states
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(device) #cell state 

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :]) 
        return out
    



class attentional_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, bidirectional=False):
        super(attentional_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        #attention layer
        self.attention = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)


        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)

    def forward(self, x):
        device = x.device
        num_directions = 2 if self.bidirectional else 1 
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(device) 

        out, _ = self.lstm(x, (h0, c0)) #lstm_output : (batch, seq_len, hidden_size)
        #add learnable attention coeffs
        attn_scores = self.attention(out) # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1) 
        context_vector = torch.sum(attn_weights * out, dim=1)  # (batch, hidden_size)

        output = self.fc(context_vector)
        return output
    