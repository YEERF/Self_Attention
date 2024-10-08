import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self,dim):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.in_proj = nn.Linear(dim, dim*3)
        self.att_drop = nn.Dropout(0.1)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, X, attention_mask = None):
        QKV = self.in_proj(X)
        Q, K, V = torch.split(QKV, self.dim, dim=-1)
        attention_value = torch.matmul(Q, K.transpose(-1, -2))
        if attention_mask is not None:
            attention_value = attention_value.masked_fill(attention_mask == 0, float("-1e20"))
        attention_weights = torch.softmax(attention_value/math.sqrt(self.dim), dim=-1)
        attention_weights = self.att_drop(attention_weights)
        output = torch.matmul(attention_weights, V)
        return self.out_proj(output)

if __name__ == '__main__':
    x = torch.randn(3,2,4)
    y = torch.tensor([[1,1],[0,1],[1,1]])
    y = y.unsqueeze(dim=1).repeat(1,2,1)
    net = SelfAttention(4)
    out = net(x,y)
    print(out.shape)

