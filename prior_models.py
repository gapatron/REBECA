import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbed(nn.Module):
    def __init__(self, dim, fourier_dim=256):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, t):
        # t: (B,) integer or float timesteps in [0, T)
        half = self.fourier_dim // 2
        freqs = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(1.0)),
                torch.log(torch.tensor(10000.0)),
                steps=half, device=t.device
            )
        )
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        return self.proj(emb)  # (B, dim)

class AdaLNZero(nn.Module):
    """Adaptive LayerNorm with zero-initialized scale/shift heads."""
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_scale = nn.Linear(cond_dim, hidden_dim)
        self.to_shift = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.to_scale.weight); nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight); nn.init.zeros_(self.to_shift.bias)
    def forward(self, x, c):
        x = self.norm(x)
        scale = self.to_scale(c).unsqueeze(1)  # (B,1,H)
        shift = self.to_shift(c).unsqueeze(1)
        return x * (1 + scale) + shift

class GatedCrossAttn(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True
        )
        self.gate = nn.Parameter(torch.zeros(1))  # start closed
    def forward(self, x, kv):
        # x: (B, S, H) query; kv: (B, C, H) key/value (conditioning)
        y, _ = self.attn(query=x, key=kv, value=kv, need_weights=False)
        return x + self.gate * y

class LearnedTokenizer(nn.Module):
    """Project 1D embedding <-> tokens instead of .view()"""
    def __init__(self, embed_dim, n_tokens):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.token_dim = embed_dim // n_tokens
        assert embed_dim % n_tokens == 0
        self.to_tokens = nn.Linear(embed_dim, embed_dim)   # (B,E)->(B,E) then reshape
        self.from_tokens = nn.Linear(embed_dim, embed_dim) # detokenizer after concat
    def split(self, x):
        z = self.to_tokens(x)  # (B,E)
        return z.view(x.size(0), self.n_tokens, self.token_dim)  # (B,T,H)
    def merge(self, tokens):
        z = tokens.reshape(tokens.size(0), -1)  # (B,E)
        return self.from_tokens(z)  # (B,E)

class PriorBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, cond_dim, mlp_mult=4):
        super().__init__()
        self.adaln1 = AdaLNZero(hidden_dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.adaln2 = AdaLNZero(hidden_dim, cond_dim)
        self.cross_attn = GatedCrossAttn(hidden_dim, n_heads)
        self.adaln3 = AdaLNZero(hidden_dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_mult*hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_mult*hidden_dim, hidden_dim),
        )
        # zero-init last linear for residual safety
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, x, cond_vec, cond_tokens):
        # cond_vec: (B, cond_dim) summarized conditioning (time+user+score)
        h = self.adaln1(x, cond_vec)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.adaln2(x, cond_vec)
        x = self.cross_attn(h, cond_tokens)        # gated cross-attn
        h = self.adaln3(x, cond_vec)
        x = x + self.mlp(h)                         # gated via zero-init
        return x

class RebecaDiffusionPrior(nn.Module):
    """
    Drop-in replacement: learned tokenizer + AdaLN-Zero + gated per-block cross-attn
    """
    def __init__(self,
                 img_embed_dim=1280,
                 num_users=210,
                 num_tokens=8,
                 hidden_dim=None,
                 n_heads=8,
                 num_layers=6,
                 score_classes=2):
        super().__init__()
        hidden_dim = hidden_dim or (img_embed_dim // num_tokens)
        assert img_embed_dim % num_tokens == 0
        self.tok = LearnedTokenizer(img_embed_dim, num_tokens)
        self.hidden_dim = hidden_dim
        self.n_tokens = num_tokens
        self.token_dim = img_embed_dim // num_tokens

        # Project token_dim -> hidden_dim (optional widening)
        self.in_proj  = nn.Linear(self.token_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.token_dim)

        # Conditioning
        self.user_emb  = nn.Embedding(num_users + 1, hidden_dim)       # last index = null user (CFG)
        self.score_emb = nn.Embedding(score_classes + 1, hidden_dim)        # include CFG/null
        self.t_embed   = TimestepEmbed(hidden_dim)

        # small MLP to summarize cond tokens -> cond_vec
        self.cond_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PriorBlock(hidden_dim, n_heads, cond_dim=hidden_dim) for _ in range(num_layers)
        ])

        # prediction head with schedule-aware scaling (see notes below)
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def build_cond(self, t, user_id, score):
        u = self.user_emb(user_id)     # (B,H)
        s = self.score_emb(score)      # (B,H)
        te = self.t_embed(t)           # (B,H)
        cond_tokens = torch.stack([u, s, te], dim=1)   # (B,3,H)
        cond_vec = self.cond_pool(cond_tokens.mean(dim=1))  # (B,H)
        return cond_vec, cond_tokens

    def forward(self, x, t, user_id, score):
        B = x.size(0)
        img_tokens = self.tok.split(x)                  # (B,T,Htok)
        img_tokens = self.in_proj(img_tokens)           # (B,T,H)
        cond_vec, cond_tokens = self.build_cond(t, user_id, score)
        h = img_tokens
        for blk in self.blocks:
            h = blk(h, cond_vec, cond_tokens)
        h = self.head(h)
        h = self.out_proj(h)                            # (B,T,Htok)
        y = self.tok.merge(h)                           # (B,E)
        return y