import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function
from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
from vit_pytorch.extractor import Extractor
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder2 import EncoderDecoder
from einops import rearrange
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# distributed

def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

def all_gather_variable_batch(t):
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    size = torch.tensor(t.shape[0], device = device, dtype = torch.long)
    sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
    dist.all_gather(sizes, size)

    sizes = torch.stack(sizes)
    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim = 0)
    gathered_tensors = [torch.empty_like(padded_t, device = device, dtype = padded_t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, padded_t)

    gathered_tensor = torch.cat(gathered_tensors)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')

    gathered_tensor = gathered_tensor[mask]
    sizes = sizes.tolist()

    return gathered_tensor, sizes


class AllGather(Function):
    @staticmethod
    def forward(ctx, x):
        assert dist.is_initialized() and dist.get_world_size() > 1
        x, batch_sizes = all_gather_variable_batch(x)
        ctx.batch_sizes = batch_sizes
        return x

    @staticmethod
    def backward(ctx, grads):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = 0)
        return grads_by_rank[rank]

all_gather = AllGather.apply

# to latents

class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)
    
vit = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 2048,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    patch_dropout = 0  # https://arxiv.org/abs/2212.00794
)

vit = Extractor(vit, return_embeddings_only = True, detach = False)

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.dim = args.dim
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.img_encoder = vit
        self.dim_latents = args.dim_latents
        # to latents
        self.dim_latents = default(self.dim_latents, self.dim)
        self.img_to_latents = EmbedToLatents(self.dim, self.dim_latents)
        self.text_to_latents = EmbedToLatents(self.dim, self.dim_latents)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, text=None, mode='train'):
        # images:torch.Size([B, 3, 224, 224])、text:torch.Size([B, 25])
        _, fc_feats = self.visual_extractor(images)
        # torch.Size([B, 49, 2048]), fc_feats:torch.Size([B, 2048])
        
        if exists(images):
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
            image_tokens = self.img_encoder(images) #(b 3 224 224)->(b 24 2048)
            # image_tokens:torch.Size([8, 2048])??
        
        if mode == 'train':
            # print()
            image_embeds, text_embeds, output = self.encoder_decoder(fc_feats, image_tokens, text, mode='forward')
            # embedding to latents
            # 唯一用途：可以返回用来计算对比损失
            text_latents = self.text_to_latents(text_embeds) #(b 512)
            image_latents = self.img_to_latents(image_embeds) #(b 512)
            # print(type(output))
            # print(output)
            return text_latents, image_latents, output
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, image_tokens, mode='sample')
            # print(type(output))
            return output
        else:
            raise ValueError

        # return text_latents, image_latents, output