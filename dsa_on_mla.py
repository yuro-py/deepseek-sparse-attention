import torch
import torch.nn as nn
import torch.nn.functional as F

D    = 512          # model hidden dim
h    = 4            # number of attention heads
hd   = D // h       # d_head = 128 -> per-head content dim
hd_r = hd // 2      # d_rope = 64  -> per-head rope dim (shared across heads)
d_c  = 128          # compressed latent dim (query and kv), d_c << D

h_i  = 4            # indexer heads (H_I), independent hyperparam
d_i  = 64           # indexer head dim (d_I)
d_ir = d_i // 2     # indexer rope dim -> RoPE applied to first half only
topk = 4            # KV entries selected per query (2048 in the real model)

def rope(x, positions):
    dim = x.shape[-1]
    theta = 10000 ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    freqs = torch.outer(positions.float(), theta)
    freqs = torch.cat([freqs, freqs], dim=-1)

    x1, x2 = x.chunk(2, dim=-1)
    x_rotated_half = torch.cat([-x2, x1], dim=-1)

    return x * freqs.cos() + x_rotated_half * freqs.sin()

def rope_partial(x, positions):
    x_pe, x_nope = x.split(d_ir, dim=-1)                 # indexer: partial RoPE
    return torch.cat([rope(x_pe, positions), x_nope], dim=-1)

# --- MLA, MQA mode (DSA instantiation) ---
W_DQ  = nn.Linear(D, d_c, bias=False)
W_UQ  = nn.Linear(d_c, h * hd, bias=False)
W_UQA = nn.Parameter(torch.randn(h, hd, d_c) * 0.02)     # q^C -> q^A: absorb latent so query dots c_kv directly
W_QR  = nn.Linear(d_c, h * hd_r, bias=False)

W_DKV = nn.Linear(D, d_c, bias=False)
W_KR  = nn.Linear(D, hd_r, bias=False)
W_UV  = nn.Parameter(torch.randn(h, d_c, hd) * 0.02)     # per-head output up-proj (value up-proj moved post-attn)

W_O   = nn.Linear(D, D, bias=False)

# --- lightning indexer ---
W_QI  = nn.Linear(d_c, h_i * d_i, bias=False)            # from query latent c_q
W_KI  = nn.Linear(D, d_i, bias=False)                    # from hidden x (single shared key)
k_norm = nn.LayerNorm(d_i)
W_W   = nn.Linear(D, h_i, bias=False)                    # per-head indexer weights w^I

cache_c_kv = None
cache_k_r  = None
cache_k_i  = None                                        # indexer's own small key cache

def forward_step(x, pos):
    global cache_c_kv, cache_k_r, cache_k_i
    B, T, _ = x.shape

    # query: [q^A ; q^R] per head
    c_q = W_DQ(x)
    q_c = W_UQ(c_q).reshape(B, T, h, hd)
    q_a = torch.einsum("bthd,hdc->bthc", q_c, W_UQA)
    q_r = rope(W_QR(c_q).reshape(B, T, h, hd_r), positions=torch.tensor([pos], device=x.device))
    q = torch.cat([q_a, q_r], dim=-1)                    # (B,T,h,d_c+hd_r)

    # kv cache: latent + rope key, shared across heads (no per-head up-proj)
    new_c_kv = W_DKV(x)
    new_k_r  = W_KR(x)
    new_k_i  = k_norm(W_KI(x))
    if cache_c_kv is None:
        cache_c_kv, cache_k_r, cache_k_i = new_c_kv, new_k_r, new_k_i
    else:
        cache_c_kv = torch.cat([cache_c_kv, new_c_kv], dim=1)
        cache_k_r  = torch.cat([cache_k_r,  new_k_r],  dim=1)
        cache_k_i  = torch.cat([cache_k_i,  new_k_i],  dim=1)

    t = cache_c_kv.shape[1]
    k   = torch.cat([cache_c_kv, rope(cache_k_r, positions=torch.arange(t, device=x.device))], dim=-1)
    k_i = rope_partial(cache_k_i, positions=torch.arange(t, device=x.device))

    # lightning indexer: top-k over index scores (causal by construction, decode-style)
    q_i = rope_partial(W_QI(c_q).reshape(B, T, h_i, d_i), positions=torch.tensor([pos], device=x.device))
    w_i = W_W(x)                                                                      # (B,T,h_i)
    I = (F.relu(q_i @ k_i.transpose(-2, -1)) * w_i.unsqueeze(-1)).sum(-2)             # (B,T,t), eq.1
    idx = I.topk(min(topk, t), dim=-1).indices                                        # (B,T,topk)

    # --- FIX: Use advanced indexing instead of torch.gather ---
    # idx is (B, T, K). We want to select from k (B, t, Dim) along dim 1.
    # k[batch_idx, idx] works via broadcasting: (B,1,1) vs (B,T,K) -> (B,T,K)
    batch_idx = torch.arange(B, device=x.device)[:, None, None]

    k_sel = k[batch_idx, idx]                   # (B, T, K, d_c + hd_r)
    v_sel = cache_c_kv[batch_idx, idx]          # (B, T, K, d_c)

    # core attention: MQA over selected entries only
    scores = (q @ k_sel.transpose(-2, -1)) / (d_c + hd_r) ** 0.5                      # (B,T,h,topk)
    weights = F.softmax(scores, dim=-1)
    o = weights @ v_sel                                                               # sum over selected s, value = c_kv
    o = torch.einsum("bthc,hcd->bthd", o, W_UV)                                       # per-head up-proj
    o = o.reshape(B, T, D)
    return W_O(o)

x1 = torch.randn(1, 1, D)
out1 = forward_step(x1, pos=0)
print(f"cache_c_kv shape: {cache_c_kv.shape} | cache_k_i shape: {cache_k_i.shape}")

x2 = torch.randn(1, 1, D)
out2 = forward_step(x2, pos=1)
print(f"cache_c_kv shape: {cache_c_kv.shape} | cache_k_i shape: {cache_k_i.shape}")

print("----------")
print(f"dense attention attends over: t entries | DSA attends over: min(topk, t) = {min(topk, cache_c_kv.shape[1])}")
print(f"DSA per-token cache is: d_c + hd_r + d_i = {d_c + hd_r + d_i} numbers/token")
