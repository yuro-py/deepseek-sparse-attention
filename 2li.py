import torch
import torch.nn as nn
import torch.nn.functional as F

h = torch.randn(1, 6, 512)
B, T, D = h.shape
H, d = 4, 64

q = nn.Linear(D, H*d, bias=False)(h).view(B,T,H,d).transpose(1,2)
# NEW = B, H, T. D
k = nn.Linear(D, H*d, bias=False)(h).view(B,T,H,d).transpose(1,2)
# NEW = B, H, T. D
w = nn.Linear(D, H, bias=False)(h).transpose(1,2).unsqueeze(-1)
# NEW = B, H, T,

scores = (w * F.relu(q @ k.transpose(-2,-1))).sum(1)
scores = scores.masked_fill(~torch.ones(T,T,dtype=torch.bool,device=h.device).tril(), float('-inf'))
idx = scores.topk(4, -1).indices
