import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.dim = dim

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(0.1)

        self.out_proj = nn.Linear(dim,dim)

    def forward(self,X,attention_mask=None):
        query = self.query_proj(X)
        key = self.key_proj(X)
        value = self.value_proj(X)
        attention_values = torch.matmul(query, key.transpose(1,2))
        attention_weights = attention_values/math.sqrt(self.dim)
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask == 0, float("-1e20"))
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        return self.out_proj(output)


if __name__ == '__main__':
    x = torch.rand(3,2,4)
    y = torch.tensor([[1, 1], [0, 1], [1, 1]])
    y = y.unsqueeze(dim=1).repeat(1, 2, 1)
    net = SelfAttention(4)
    out = net(x, y)
    print(out.shape)





