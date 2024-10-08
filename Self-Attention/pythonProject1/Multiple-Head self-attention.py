import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,X, attention_mask=None):
        batch_size, seq_len, _ = X.size()

        query = self.query_proj(X)
        key = self.key_proj(X)
        value = self.value_proj(X)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        attention_value = torch.matmul(query, key.transpose(-1, -2))
        attention_weights = attention_value/math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_weights = attention_value.masked_fill(attention_mask == 0, float("-1e20"))

        attention_weights = torch.softmax(attention_weights, dim=3)
        attention_weights = self.dropout(attention_weights)

        mid_output = torch.matmul(attention_weights, value)
        mid_output = mid_output.transpose(1,2).contiguous()
        output = mid_output.view(batch_size, seq_len, -1)
        output = self.out_proj(output)
        return output

if __name__ == '__main__':
    x = torch.randn(3,2,128)
    y = torch.tensor([[1,0],[0,1],[1,1]])
    y = y.unsqueeze(1).unsqueeze(2).repeat(1,8,2,1)
    net = MultiHeadSelfAttention(8,128)
    output = net(x,y)
    print(output)







