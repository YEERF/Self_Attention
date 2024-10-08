import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self,dim):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.in_proj = nn.Linear(dim, dim*3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, X):
        QKV = self.in_proj(X)
        Q, K, V = torch.split(QKV, self.dim, dim=-1)
        attention_weights = torch.softmax(torch.matmul(Q,K.transpose(-1, -2))/math.sqrt(self.dim), dim=-1)
        output = torch.matmul(attention_weights, V)
        return self.out_proj(output)

if __name__ == '__main__':
    x = torch.randn(3,2,4)
    net = SelfAttention(4)
    output = net(x)
    print(output)