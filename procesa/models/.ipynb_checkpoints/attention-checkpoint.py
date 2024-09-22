import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_attention(q, k, v, dropout=None, mask=None):
    """
    :param q: Query [num_heads, 1, latent_dim]
    :param k: Key [bs, num_heads, seq_len, latent_dim]
    :param v: Value [bs, num_heads, seq_len, latent_dim]
    :param mask: [bs, seq_len]
    """
    if q.ndim + 1 == k.ndim:
        # nij: [num_heads, 1, latent_dim]
        # bnkj: [bs, num_heads, seq_len, latent_dim]
        # bnik: [bs, num_heads, 1, seq_len]        
        score = torch.einsum('nij,bnkj->bnik', q, k)
    elif q.ndim == k.ndim:
        score = torch.einsum('bnij,bnkj->bnik', q, k)
    score = score / np.sqrt(q.shape[-1])
    if mask is not None:
        mask = mask[:, None, None]
        score = score * mask + (-1e8) * (1 - mask)
    score = F.softmax(score, dim=-1)  # [bs, num_heads, 1, seq_len]
    if dropout is not None:
        score = dropout(score)
    # bnij: [bs, num_heads, 1, seq_len]
    # bnjk: [bs, num_heads, seq_len, latent_dim]
    # bnik: [bs, num_heads, 1, latent_dim]
    return torch.einsum('bnij,bnjk->bnik', score, v)  # [B, NH, NQ, EL]


class MultiHeadedAttentionBase(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        """
        :param embed_dim: The dimension of feature in each entity
        :param num_heads: The number of attention heads
        """
        super().__init__()
        self.w_k = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self.w_v = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self.w_o = nn.Parameter(torch.empty(num_heads, latent_dim, embed_dim))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)
        if hasattr(self, 'q'):
            nn.init.xavier_normal_(self.q)
        if hasattr(self, 'w_q'):
            nn.init.xavier_normal_(self.w_q)


# ManiSkill-Learn
class AttentionPooling(MultiHeadedAttentionBase):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        super().__init__(embed_dim, num_heads, latent_dim, dropout)
        self.q = nn.Parameter(torch.empty(num_heads, 1, latent_dim))
        self._reset_parameters()

    def forward(self, x, mask=None):
        """
        :param x: [bs * seq_len, embed_dim]
        :param mask: [bs, 1, seq_len]
        :return: [bs, embed_dim]
        """
        k = torch.einsum('blj,njd->bnld', x, self.w_k)  # [bs, num_heads, seq_len, latent_dim]
        v = torch.einsum('blj,njd->bnld', x, self.w_v)  # [bs, num_heads, seq_len, latent_dim]

        out = compute_attention(self.q, k, v, self.dropout, mask)  # [bs, num_heads, 1, latent_dim]

        # bnlj: [bs, num_heads, 1, latent_dim]
        # njk: [num_heads, latent_dim, embed_dim]
        # blk: [bs, 1, embed_dim]
        out = torch.einsum('bnlj,njk->blk', out, self.w_o)
        out = out[:, 0]  # [bs, embed_dim]
        return out

if __name__ == '__main__':
    layer = AttentionPoolingFlat(embed_dim=1024, num_heads=4, latent_dim=128)
    x = torch.randn((8, 20, 1024))
    out = layer(x)
    print(out.shape)

