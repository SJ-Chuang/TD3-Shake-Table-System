import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMStacked(nn.Module):
    def __init__(self, config):
        super(LSTMStacked, self).__init__()
        self.lstm = nn.LSTM(config["input_dim"], config["hidden_dim"], num_layers=config["n_lstm_layer"], batch_first=True)

        dense = []

        for _ in range(config["n_dense_layer"]):
            dense.append(nn.Linear(config["hidden_dim"], config["hidden_dim"]))
            dense.append(nn.ReLU())

        self.dense = nn.Sequential(*dense)

        self.output = nn.Linear(config["hidden_dim"], config["output_dim"])

    def forward(self, inputs, hidden):
        lstm_out, hidden_state = self.lstm(inputs, hidden)
        dense_out = self.dense(lstm_out)
        output = self.output(dense_out)
        
        return output, hidden_state