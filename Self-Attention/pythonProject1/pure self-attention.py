import torch
import torch.nn as nn
import math

import warnings
warnings.filterwarnings("ignore")

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hiddden_dim = hidden_dim

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        attention_value = torch.matmul(Q, K.transpose(1, 2))
        attention_weight = torch.softmax(attention_value/math.sqrt(self.hiddden_dim), dim=-1)

        output = torch.matmul(attention_weight, V)
        return output

if __name__ == '__main__':
    x = torch.randn(3,2,4)
    net = SelfAttention(4)
    output = net(x)
    print(output)