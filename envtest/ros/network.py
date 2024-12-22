import torch
import torch.nn as nn
import numpy as np


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * \
            (img_size[1] // patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  # Since both kernel size and stride are equal to patch_size, there will never be overlap when kernel slides along the input tensor and fall exactly in line with patches we have in mind

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape '(n_samples, in_chans, img_size, img_size)' (bunch of images where n_samples = batch size)

        Returns
        -------
        torch.Tensor of shape '(n_samples, n_patches, embed_dim)'
        """
        x = self.proj(
            x
        )  # '(n_samples, embed_dim, n_patches**0.5, n_patches**0.5)'
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # '(n_samples, n_patches, embed_dim)'

        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads ==
                embed_size), "Embed size needs to be div by heads"
        
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads*self.head_dim, embed_size)

    def forward(self, decoder_values, encoder_keys, encoder_values, mask):
        N = decoder_values.shape[0]
        decoder_len, encoder_len = decoder_values.shape[1], encoder_keys.shape[1]
        
        values = self.values(encoder_values)
        keys = self.keys(encoder_keys)
        values = values.reshape(N, encoder_len, self.heads, self.head_dim)
        keys = keys.reshape(N, encoder_len, self.heads, self.head_dim)
        queries = self.queries(decoder_values)
        queries = queries.reshape(N, decoder_len, self.heads, self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, decoder_len, self.heads*self.head_dim)
        out = self.fc_out(out)
        return out

class SelfAttention(nn.Module):  # Inheriting from nn.Module
    # heads is the number of ways we're dividing the embed into
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()  # Initializing the parent class
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Ensure that the input is cleanly divisible by given head size
        assert (self.head_dim * heads ==
                embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        # Fully_connected_out, after concat - so heads*head_dim should be = to embed size
        self.fc_out = nn.Linear(self.heads*self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        # No. of training examples, examples sent in at the same time
        N = queries.shape[0]
        # these lens are going to be equal to either the source sentence len or target sentence len depending on where we use encoder, so we abstract them to store vals
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.heads pieces
        values = self.values(values)
        keys = self.keys(keys)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = self.queries(queries)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Used for matmul where we have several other dims
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape = (N, query_len, heads, heads_dim)
        # keys shape = (N, key_len, heads, heads_dim)
        # energy shape = (N, heads, query_len, key_len)

        if mask is not None:
            # if element of mask is 0, we want to shut that off so it doesn't impact any other; mask for target is gonna be a triangular matrix and element where it closes is 0
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # when we have -inf values, they'll be set to 0, this formula is from paper, dim=3 means we're normalizing across the key_len, which would denote the source/ target sentence depending on where this is called
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum(
            "nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after einsum: (N, query_len, heads, heads_dim) -> then flatten last two dimensions (concat)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = CrossAttention(embed_size, heads)
        # Layer norm is like batch norm but takes norm per example instead of whole batch; more computation
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            # Mapping it to some more nodes related to forward expansion
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            # This feed forward network won't change anything, just computation and change it back to original
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))  # Skip connection

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))

        return out


class Transformer(nn.Module):
    def __init__(
            self,
            img_size,
            patch_size,
            in_chans,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
    ):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, 1+self.patch_embed.n_patches, embed_size))
        self.norm = nn.LayerNorm(embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, 3)

    def forward(self, x, mask):
        N = x.shape[0]

        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            N, -1, -1
        )
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        out = self.dropout(x)

        for layer in self.layers:
            # Since we are in encoder, value, key and query are all gonna be the same
            out = layer(out, out, out, mask)

        out = self.norm(out)
        out = self.fc_out(out[:, 0])
        # out = torch.tanh(out)

        return out[0]
