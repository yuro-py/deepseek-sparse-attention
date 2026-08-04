import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, D = 2, 8, 512

x = torch.randn(B, T, D)

d_i = 64
k = 4

W_qi = nn.Linear(D, d_i, bias=False)
W_ki = nn.Linear(D, d_i, bias=False)
W_w  = nn.Linear(D, 1, bias=False)

q_i = W_qi(x)
k_i = W_ki(x)
w = W_w(x).squeeze(-1)

scores = q_i @ k_i.transpose(-1, -2)

scores = scores + w.unsqueeze(1)

scores = F.relu(scores)

values, indices = scores.topk(k, dim=-1)
