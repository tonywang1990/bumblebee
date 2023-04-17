import os
import numpy as np
import pickle
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional
from tqdm import tqdm
import tiktoken
import sentencepiece as spm


@dataclass
class GPTConfig:
    batch_size: int = 16  # B
    block_size: int = 128  # T
    n_emb: int = 4*8  # E
    n_head: int = 4
    n_layer: int = 3
    vocab_size: int = 1334 # C
    num_epoch: int = 2000
    learning_rate: float = 3e-4
    dropout: float = 0.2
    device = 'cpu'


class Data(object):
    def __init__(self, config: GPTConfig):
        self.config = config
        #self.load_lunyu()
        #self.load_shakespeare()
        self.load_shakespeare_char()

    def load_shakespeare_char(self):
        data_dir = "data/shakespeare_char"
        self.train_data = np.memmap(os.path.join(
            data_dir, 'train.bin'), dtype=np.uint16, mode='r').astype(int)
        self.val_data = np.memmap(os.path.join(
            data_dir, 'val.bin'), dtype=np.uint16, mode='r').astype(int)
        meta = pickle.load(open(os.path.join(data_dir, 'meta.pkl'), 'rb'))
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        self.lgiirlflifngliccubhegkkuknfeckbjbfdtlvhktrhikjlgcgdkkcitos = meta['itos']
        self.vocab_size = meta['vocab_size']
        self.encode = lambda x: [self.stoi[c] for c in x]
        self.decode = lambda x: ''.join([self.itos[i] for i in x])
    
    def load_shakespeare(self):
        data_dir = "data/shakespeare"
        self.train_data = np.memmap(os.path.join(
            data_dir, 'train.bin'), dtype=np.uint16, mode='r').astype(int)
        self.val_data = np.memmap(os.path.join(
            data_dir, 'val.bin'), dtype=np.uint16, mode='r').astype(int)
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.max_token_value+1
        self.encode=self.enc.encode_ordinary
        self.decode=self.enc.decode
    
    def load_lunyu(self):
        data_dir = "data/lunyu"
        self.train_data = np.memmap(os.path.join(
            data_dir, 'train.bin'), dtype=np.uint16, mode='r').astype(int)
        self.val_data = np.memmap(os.path.join(
            data_dir, 'val.bin'), dtype=np.uint16, mode='r').astype(int)

        # train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
        # `m.vocab` is just a reference. not used in the segmentation.
        spm.SentencePieceTrainer.train(f'--input=data/lunyu/output.txt --model_prefix=lunyu --vocab_size={self.config.vocab_size}')
        # makes segmenter instance and loads the model file (m.model)
        self.enc = spm.SentencePieceProcessor()
        self.enc.load(os.path.join(data_dir, 'lunyu.model'))
        #self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.config.vocab_size #self.enc.max_token_value+1
        self.encode = self.enc.encode_as_ids
        self.decode = self.enc.decode_ids

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == 'train':
            data = torch.from_numpy(np.array(self.train_data))
        elif split == 'val':
            data = torch.from_numpy(np.array(self.val_data))
        else:
            raise NotImplementedError
        ids = torch.randint(low=0, high=len(
            data)-self.config.block_size, size=(self.config.batch_size,))
        x = torch.stack([data[i: i + self.config.block_size] for i in ids])
        y = torch.stack([data[i+1: i+self.config.block_size+1] for i in ids])
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y


class Head(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        n_emb, n_head, block_size, dropout = config.n_emb, config.n_head, config.block_size, config.dropout
        head_size = n_emb // n_head
        self.query = nn.Linear(n_emb, head_size)
        self.key = nn.Linear(n_emb, head_size)
        self.value = nn.Linear(n_emb, head_size)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # input: (B, T, E)
        B, T, E = emb.shape
        q = self.query(emb)  # q,k,v = (B, T, H)
        k = self.key(emb)
        v = self.value(emb)
        weight = q @ k.transpose(-2, -1) * (E**-0.5)  # (B, T, T)
        # As a decoder block, mask out future information
        weight = weight.masked_fill(self.tril[:T, :T] == 0, -torch.inf)
        weight = torch.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        out = weight @ v  # (B, T, H)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        n_emb, n_head = config.n_emb, config.n_head
        self.heads = nn.ModuleList([Head(config) for _ in range(n_head)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(input) for h in self.heads], dim=-1)
        x = self.dropout(self.proj(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        n_emb = config.n_emb
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(config.dropout)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        n_emb = config.n_emb
        self.sa_head = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # residual connection, pre-norm
        x = input + self.sa_head(self.ln1(input))
        x = x + self.feed_forward(self.ln2(x))  # residual connection, pre-norm
        return x


class BigramModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(
            config.vocab_size, config.n_emb)
        self.position_embedding_table = nn.Embedding(
            config.block_size, config.n_emb)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)],
            nn.LayerNorm(config.n_emb)
        )
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # idx shape = [B, T]
        idx = idx[:, -self.config.block_size:]
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)  # [B, T, E]
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.config.device))  # [T, E]
        x = token_emb + pos_emb  # [B, T, E]
        x = self.blocks(x)  # [B, T, H=E]
        logits = self.lm_head(x)

        if targets is not None:
            logits = logits.view(self.config.batch_size *
                                 self.config.block_size, self.config.vocab_size)
            targets = targets.view(
                self.config.batch_size * self.config.block_size)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss  # pyre-ignore[7]

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self.forward(
                idx[:, -self.config.block_size:])  # [B, T ,C]
            logits = logits[:, -1, :]  # [B, C]
            probs = torch.softmax(logits, dim=-1)  # [B, C]
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat((idx, idx_next), dim=-1)  # [B, C+1]
        return idx


class GPT(object):
    def __init__(self, config: GPTConfig):
        self.data = Data(config)
        self.config = config
        self.config.vocab_size = self.data.vocab_size
        self.model = BigramModel(config)

    @torch.no_grad()
    def estimate_loss(self, eval_iters: int = 10):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                input, targets = self.data.get_batch(split)
                _, loss = self.model(input, targets)
                losses[i] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate)
        for i in tqdm(range(self.config.num_epoch)):
            input, targets = self.data.get_batch('train')
            logits, loss = self.model(input, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if i % (self.config.num_epoch/10) == 0:
                losses = self.estimate_loss()
                print(losses)

    def generate(self, length: int):
        output = self.model.generate(torch.zeros(
            (1, 1), dtype=torch.long, device=self.config.device), length)[0].tolist()
        return self.data.decode(output)

if __name__ == '__main__':
    config = GPTConfig()
    gpt = GPT(config)
    gpt.train()
    print(gpt.generate(1000))
