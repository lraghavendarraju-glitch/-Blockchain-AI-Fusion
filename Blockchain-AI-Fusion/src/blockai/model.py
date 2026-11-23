"""Native-PyTorch GATv2 implementation; no compiled torch-geometric dependency.

Exact formula: e_ij = a^T LeakyReLU(W_src x_i + W_dst x_j).
Softmax and message aggregation are performed over destination-node incoming edges.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from .features import GraphSample


@dataclass
class GraphBatch:
    numeric: torch.Tensor
    type_ids: torch.Tensor
    role_ids: torch.Tensor
    edge_index: torch.Tensor
    global_features: torch.Tensor
    labels: torch.Tensor
    step_labels: torch.Tensor
    lengths: list[int]
    workflow_ids: list[str]
    step_ids: list[list[str]]

    def to(self, device: torch.device | str) -> "GraphBatch":
        for key in ("numeric","type_ids","role_ids","edge_index","global_features","labels","step_labels"):
            setattr(self, key, getattr(self,key).to(device))
        return self


def collate_graphs(samples: Sequence[GraphSample]) -> GraphBatch:
    lengths = [len(s.numeric) for s in samples]
    offset = 0; edges = []
    for sample, n in zip(samples,lengths):
        if sample.edge_index.shape[1] > 0:
            edges.append(torch.as_tensor(sample.edge_index, dtype=torch.long) + offset)
        offset += n
    return GraphBatch(
        numeric=torch.cat([torch.tensor(s.numeric,dtype=torch.float32) for s in samples],0),
        type_ids=torch.cat([torch.tensor(s.type_ids,dtype=torch.long) for s in samples]),
        role_ids=torch.cat([torch.tensor(s.role_ids,dtype=torch.long) for s in samples]),
        edge_index=torch.cat(edges,dim=1) if edges else torch.empty(2,0,dtype=torch.long),
        global_features=torch.stack([torch.tensor(s.global_features,dtype=torch.float32) for s in samples]),
        labels=torch.tensor([s.label for s in samples],dtype=torch.float32),
        step_labels=torch.cat([torch.tensor(s.step_labels,dtype=torch.float32) for s in samples]),
        lengths=lengths, workflow_ids=[s.workflow_id for s in samples],
        step_ids=[s.step_ids for s in samples],
    )


class GATv2Layer(nn.Module):
    def __init__(self, in_dim: int = 128, n_heads: int = 4, head_dim: int = 32, dropout: float = .3):
        super().__init__()
        self.heads = n_heads
        self.dim = head_dim
        self.proj_src = nn.Linear(in_dim, n_heads*head_dim, bias=False)
        self.proj_dst = nn.Linear(in_dim, n_heads*head_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(n_heads, head_dim))
        nn.init.xavier_uniform_(self.attention)
        self.bias = nn.Parameter(torch.zeros(n_heads*head_dim))
        self.norm = nn.LayerNorm(n_heads*head_dim)
        self.residual = nn.Linear(in_dim, n_heads*head_dim) if in_dim != n_heads*head_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(.2)
        self.output_act = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n=x.shape[0]
        selfloop = torch.arange(n,device=x.device)
        src=torch.cat([edge_index[0],selfloop]); dst=torch.cat([edge_index[1],selfloop])
        hsrc=self.proj_src(x).view(n,self.heads,self.dim)
        hdst=self.proj_dst(x).view(n,self.heads,self.dim)
        logits=(self.act(hsrc[src]+hdst[dst])*self.attention).sum(-1)  # [E,heads]
        index=dst[:,None].expand(-1,self.heads)
        max_values=torch.full((n,self.heads),-1e9,device=x.device,dtype=x.dtype)
        max_values=max_values.scatter_reduce(0,index,logits,reduce="amax",include_self=True)
        weights=torch.exp(logits-max_values[dst])
        denominator=torch.zeros((n,self.heads),device=x.device,dtype=x.dtype)
        denominator.index_add_(0,dst,weights)
        alpha=self.dropout(weights/denominator[dst].clamp_min(1e-9))
        messages=(hsrc[src]*alpha.unsqueeze(-1)).reshape(-1,self.heads*self.dim)
        agg=torch.zeros((n,self.heads*self.dim),device=x.device,dtype=x.dtype)
        agg.index_add_(0,dst,messages)
        return self.norm(self.output_act(agg+self.bias)+self.residual(x))


class ProvSeqGAT(nn.Module):
    def __init__(self, gru_hidden: int = 64, gat_heads: int = 4, gat_layers: int = 2,
                 dropout: float = .3, n_types: int = 13, n_roles: int = 9,
                 use_gru: bool = True, use_integrity: bool = True,
                 use_actor: bool = True, use_temporal: bool = True):
        super().__init__()
        if gat_heads not in (2,4,8) or 128 % gat_heads:
            raise ValueError("gat_heads must divide 128")
        self.use_gru=use_gru
        self.use_integrity=use_integrity
        self.use_actor=use_actor
        self.use_temporal=use_temporal
        self.type_emb=nn.Embedding(n_types,16,padding_idx=0)
        self.role_emb=nn.Embedding(n_roles,8,padding_idx=0)
        self.project=nn.Sequential(nn.Linear(64,128), nn.LayerNorm(128), nn.GELU())
        self.bigru=nn.GRU(128,gru_hidden,batch_first=True,num_layers=2,bidirectional=True,
                          dropout=dropout)
        self.gru_to_graph=nn.Linear(gru_hidden*2,128) if gru_hidden != 64 else nn.Identity()
        self.gat=nn.ModuleList([GATv2Layer(128,gat_heads,128//gat_heads,dropout) for _ in range(gat_layers)])
        self.attention_pool=nn.Linear(128,1)
        self.head=nn.Sequential(nn.Linear(134,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,1))
        self.step_head=nn.Linear(128,1)

    def forward(self, batch: GraphBatch) -> dict[str,torch.Tensor]:
        numeric=batch.numeric.clone()
        if not self.use_actor: numeric[:,20:24]=0
        if not self.use_integrity: numeric[:,24:32]=0
        if not self.use_temporal: numeric[:,32:40]=0
        role_embedding=self.role_emb(batch.role_ids) if self.use_actor else torch.zeros(
            (len(batch.role_ids),8),device=numeric.device,dtype=numeric.dtype)
        x=torch.cat([self.type_emb(batch.type_ids),
                     numeric[:,:20], role_embedding, numeric[:,20:]],dim=-1)
        if x.shape[-1] != 64:
            raise AssertionError("Input feature width must be 64")
        x=self.project(x)
        chunks=x.split(batch.lengths,dim=0)
        padded=pad_sequence(chunks,batch_first=True)
        packed=pack_padded_sequence(padded,batch.lengths,batch_first=True,enforce_sorted=False)
        if self.use_gru:
            seq,_=self.bigru(packed)
            padded,_=pad_packed_sequence(seq,batch_first=True)
            x=torch.cat([padded[i,:n] for i,n in enumerate(batch.lengths)],0)
            x=self.gru_to_graph(x)
        for layer in self.gat:
            x=layer(x,batch.edge_index)
        step_logits=self.step_head(x).squeeze(-1)
        attention_logits=self.attention_pool(x).squeeze(-1)
        pooled=[]; start=0
        for length in batch.lengths:
            att=torch.softmax(attention_logits[start:start+length],dim=0)
            pooled.append((x[start:start+length]*att[:,None]).sum(dim=0))
            start+=length
        pooled=torch.stack(pooled)
        wf_logits=self.head(torch.cat([pooled,batch.global_features],dim=1)).squeeze(-1)
        return {"workflow_logits":wf_logits,"step_logits":step_logits}

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
