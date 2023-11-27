from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.autograd import Function
import torch.distributed as dist
from einops import rearrange, repeat
from .att_model import pack_wrapper, AttModel
from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
from vit_pytorch.extractor import Extractor

# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

def matrix_sys(matrix):
    matrix = matrix.detach().numpy()
    if np.transpose(matrix).all() == matrix.all():
        return True
    else:
        return False

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1) # d_model/head = 512/8 = 64
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 2.softmax进行归一化
    p_attn = F.softmax(scores, dim=-1) # [B,head,98,98]/[B,head,118,118]
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)
    
def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)
    
def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# to latents

class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)

# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = cocaLayerNorm(dim)
        self.context_norm = cocaLayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)# torch.Size([8, 257, 512])
        context = self.context_norm(context)# torch.Size([8, 49, 512])

        # get queries
        # q:torch.Size([8, 257, 64])
        q = self.to_q(x)
        # q:torch.Size([8, 8, 257, 64])
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale
        # q:torch.Size([8, 8, 257, 64])
        q = q * self.scale
        
        # get key / values
        # k:torch.Size([8, 49, 64])、v:torch.Size([8, 49, 64])
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity
        # sim:torch.Size([8, 8, 257, 49])
        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention
        # sim:torch.Size([8, 8, 257, 49])、attn:torch.Size([8, 8, 257, 49])
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate
        # out:torch.Size([8, 8, 257, 64])
        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)
        # out:torch.Size([8, 257, 512])
        return out


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, encoder,unimodal_layers, decoder, tgt_embed, token_emb, text_cls, d_model):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.unimodal_layers = unimodal_layers
        self.tgt_embed = tgt_embed
        self.token_emb = token_emb
        # self.rm = rm
        self.d_model = d_model
        self.text_cls_token = text_cls
        self.text_cls_norm = cocaLayerNorm(d_model)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # src(att_feats):      [B,49,512]-[b 24 512] image_tokens
        # tgt(seq):            [B,seq_length-1] text
        # src_mask(att_masks): [B,1,49]
        # tgt_mask(seq_masks): [B,seq_length-1,seq_length-1]
        batch = tgt.shape[0]
        text_tokens = self.token_emb(tgt) #(b len)->(b len 512)

        # append text cls tokens
        text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch) #(b 1 512)
        text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2) #(b len+1 512)

        image_embeds, image_tokens = self.encode(src, src_mask) #(b 512) (b 256 512)
        text_embeds, text_tokens = self.decode(image_tokens, src_mask, text_tokens, tgt_mask) #(b 512) (b seq 512)
        
        return image_embeds, text_embeds, text_tokens

    def encode(self, src, src_mask):
        image_embeds, image_tokens = self.encoder(src, src_mask) 
        # (b 512), (b 49 512)-(b 256 512)
        return image_embeds, image_tokens

    # tgt == ys
    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        # memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        # memory = self.rm(self.tgt_embed(tgt), memory)
        # tgt_embed: (b seq)->(b seq 512)
        # 这部分与文本生成相关，原代码用了一个tgt_embed，但是这里的tgt_embed和原来的不一样
        if tgt.shape[-1] != 512:
            tgt = self.tgt_embed(tgt)
            # tgt = torch.unsqueeze(tgt, -1)
            # tgt = tgt.repeat(1, 1, 512)

        # use unimodal_layers
        for attn_ff in self.unimodal_layers:
            tgt = attn_ff(tgt, attn_mask=tgt_mask)

        text_cls_tokens = tgt[:, -1]
        text_embeds = self.text_cls_norm(text_cls_tokens)

        # text_embeds, text_tokens = self.decoder(tgt, hidden_states, src_mask, tgt_mask) #(b seq 512)
        text_tokens = self.decoder(tgt, hidden_states, src_mask, tgt_mask)#(b seq 512)
        # text_tokens = self.to_logits(text_tokens)
        # for attn_ff, cross_attn in self.multimodal_layers:
        #     text_tokens = attn_ff(text_tokens)
        #     text_tokens = cross_attn(text_tokens, hidden_states)
        # text_tokens, text_cls_tokens = result[:, :-1], result[:, -1]
        # text_embeds = self.text_cls_norm(text_cls_tokens)
        #(b 512), (b seq 512)
        return text_embeds, text_tokens


class ImageEncoder(nn.Module):
    def __init__(self, img_queries, img_attn_pool, img_attn_pool_norm):
        super(ImageEncoder, self).__init__()
        self.img_queries = img_queries
        self.img_attn_pool = img_attn_pool
        self.img_attn_pool_norm = img_attn_pool_norm

    def forward(self, image_tokens, image_mask=None):
        # image_tokens(b 24 2048)
        # attention pool image tokens
        img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens.shape[0]) #(b 257 512)
        img_queries = self.img_attn_pool(img_queries, image_tokens) #(b 257 512) CrossAttention
        img_queries = self.img_attn_pool_norm(img_queries) #(b 257 512)
        image_embeds, image_tokens = img_queries[:, 0], img_queries[:, 1:]
        
        # (b 512)计算loss (b 256 512)传入decoder
        return image_embeds, image_tokens

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.float().mean(-1, keepdim=True)
        std = x.float().std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# normalization
# they use layernorm without bias, something that pytorch does not offer

class cocaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        # x:torch.Size([8, 257, 512])
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.mask = None
        self.pos_emb = None

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n].to(device)

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.mask = mask
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n].to(device)

        pos_emb = self.rotary_emb(n, device=device)
        self.pos_emb = pos_emb
        return pos_emb

    def forward(self, x, attn_mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        # x:(b len+1 512)
        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x) # x:(b len+1 512)

        # attention queries, keys, values, and feedforward inner
        # q(b len+1 512) k(b len+1 64) v(b len+1 64) ff(b len+1 4096)
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h) #(b 8 len+1 64)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device) #(len+1 64)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity(b 8 i 64, b j 64 -> b 8 i j)

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device) #(n,n)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # extra attention mask - for masking out attention from text CLS token to padding

        if exists(attn_mask):
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)

# class Decoder(nn.Module):
#     def __init__(self, layer, N):
#         super(Decoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.d_model)

#     def forward(self, x, hidden_states, src_mask, tgt_mask):
#         for layer in self.layers:
#             # text_embeds, x = layer(x, hidden_states, src_mask, tgt_mask)
#             x = layer(x, hidden_states, src_mask, tgt_mask)
#         return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        for layer in self.layers:
            # text_embeds, x = layer(x, hidden_states, src_mask, tgt_mask)
            x = layer(x, hidden_states, src_mask, tgt_mask)
            # text_tokens = attn_ff(x)
            # text_tokens = cross_attn(text_tokens, hidden_states)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
        

    def forward(self, x, hidden_states, src_mask, tgt_mask):

        # 假设你的图像特征是 image_features，形状为 [batch size, 256, 512]
        batch_size, height, width = hidden_states.size(0), hidden_states.size(1), hidden_states.size(2)
        reshaped_features = hidden_states.view(batch_size, -1, 512)  # 将第二维重塑为 49
        reshaped_features = reshaped_features[:, :49, :]  # 切片，保留前 49 个时间步!!!

        # reshaped_features 现在的形状为 [batch size, 49, 512]

        m = reshaped_features
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) #(b seq 512)
        # text_cls_tokens = x[:, -1] #(b 512)
        # text_embeds = self.text_cls_norm(text_cls_tokens)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) # (b seq 512)
        # return text_embeds, self.sublayer[2](x, self.feed_forward)
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # [B,8,118,64] [B,8,118,118]

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # word embedding + position embedding
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        # image encoder
        img_queries = nn.Parameter(torch.randn(self.num_img_queries + 1, self.dim)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        img_attn_pool = CrossAttention(dim=self.dim, context_dim=self.image_dim, dim_head=self.dim_head, heads=self.heads, norm_context=True)
        img_attn_pool_norm = cocaLayerNorm(self.dim)
        text_cls = nn.Parameter(torch.randn(self.dim))
        token_emb = nn.Embedding(tgt_vocab, self.dim)
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        unimodal_layers = nn.ModuleList([])
        for ind in range(self.unimodal_depth):
            unimodal_layers.append(
                Residual(ParallelTransformerBlock(dim=self.dim, dim_head=self.dim_head, heads=self.heads, ff_mult=self.ff_mult)),
            )
        multimodal_layers = nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=self.dim, dim_head=self.dim_head, heads=self.heads, ff_mult=self.ff_mult)),
                Residual(CrossAttention(dim=self.dim, dim_head=self.dim_head, heads=self.heads, parallel_ff=True, ff_mult=self.ff_mult))
            ])
        # they used embedding weight tied projection out to logits, not common, but works
        # to_logits[-1].weight = token_emb.weight
        # nn.init.normal_(token_emb.weight, std=0.02)
        # # position = PositionalEncoding(self.d_model, self.dropout)
        # rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = Transformer(
            ImageEncoder(c(img_queries), c(img_attn_pool), c(img_attn_pool_norm)),
            unimodal_layers,
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            # Decoder(multimodal_layers,self.multimodal_depth),
            # multimodal_layers,
            Embeddings(self.d_model, tgt_vocab),
            # nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            token_emb,
            text_cls,
            self.d_model
            )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.dim = args.dim
        self.image_dim = args.image_dim
        self.dim_head = args.dim_head
        self.heads = args.heads
        self.num_img_queries = args.num_img_queries
        self.unimodal_depth = args.unimodal_depth
        self.multimodal_depth = args.multimodal_depth
        self.ff_mult = 4
        # self.rm_num_slots = args.rm_num_slots
        # self.rm_num_heads = args.rm_num_heads
        # self.rm_d_model = args.rm_d_model

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        _, memory = self.model.encode(att_feats, att_masks) #(b 49 512)
        batch_size, height, width = memory.size(0), memory.size(1), memory.size(2)
        memory = memory.view(batch_size, -1, 512)  # 将第二维重塑为 49
        memory = memory[:, :49, :]  #为什么只要49

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks #memory=image_tokens

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)     #!! att_feats[B,49,2048]-[b 24 2048]
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks) #!! att_feats[B,49,512]-[b 24 512]
        # [B 49 512]

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long) # [B,49]
            # att_masks:torch.Size([8, 1, 49])
        att_masks = att_masks.unsqueeze(-2) # [B,1,49]-[b 1 24]!!
        # att_masks:torch.Size([8, 1, 49])

        if seq is not None:
            # crop the last one
            # seq = seq[:, :-1] #[B,seq_len-1]
            # seq_mask = (seq.data > 0)
            # seq_mask[:, 0] += True
            
            len = seq.shape[1]
            cls_mask = rearrange(seq!=0, 'b j -> b 1 j') #(b 1 len)
            seq_mask = F.pad(cls_mask, (0, 1, len, 0), value=True) #(b len+1 len+1)

            # seq_mask = seq_mask.unsqueeze(-2)
            # seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None
        #[B,49,512], [B,seq-1], [B,1,49], [B,seq-1,seq-1]
        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        # att_feats[B,49,2048]-[b 24 2048]
        # seq[B,seq]
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        # att_feats[B,49,512]-[b 24 512] # att_masks[B,1,49]-[b 1 24]其实用不到
        # seq[B,seq-1]  # seq_masks[B,seq-1,seq-1]

        image_embeds, text_embeds, out = self.model(att_feats, seq, att_masks, seq_mask) # [B,seq,512]
        outputs = F.log_softmax(self.logit(out), dim=-1) #[B,seq,vocab]
        return image_embeds, text_embeds, outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        _, out= self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
