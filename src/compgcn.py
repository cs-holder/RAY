# from src.utils.ops import *
from src.utils import get_param, scatter_add, ccorr, scatter_mean

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter



def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'max']

    # op = getattr(torch_scatter, 'scatter_{}'.format(name))
    if name == "add":
        op = scatter_add
    elif name == "mean":
        op = scatter_mean
    else:
        raise ValueError("Decoder must be chosen in [add, mean]")
    fill_value = -1e38 if name == 'max' else 0
    out = op(src, index, 0, None, dim_size, fill_value)
    if isinstance(out, tuple):
        out = out[0]

    if name == 'max':
        out[out == fill_value] = 0

    return out


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    """

    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()

    def propagate(self, aggr, edge_index, edge_type, **kwargs):        # kwargs: x, edge_type, rel_embed, edge_norm, mode
        r"""The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""

        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index               # kwargs: x, edge_type, rel_embed, edge_norm, mode, edge_index

        size = kwargs['ent_embed'].size(0)
        message_args = [edge_index, edge_type, kwargs['query_type'], kwargs['ent_embed'], kwargs['rel_embed'], 
                        kwargs['edge_norm'], kwargs['mode']]

        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0] % size, dim_size=size)  # Aggregated neighbors for each vertex
        # out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out



class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, act=lambda x: x, params=None):
        super(self.__class__, self).__init__()

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        self.w_loop = get_param((in_channels, out_channels))
        self.w_out = get_param((in_channels, out_channels))
        self.w_rel = get_param((in_channels, out_channels))
        self.loop_rel = get_param((1, in_channels))
        
        self.act = nn.LeakyReLU()
        self.W = nn.Linear(in_channels, in_channels)
        nn.init.xavier_normal_(self.W.weight.data)
        self.a = nn.Linear(in_channels * 2, 1)
        nn.init.xavier_normal_(self.a.weight.data)

        self.drop = torch.nn.Dropout(self.p.hid_drop)
        self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.p.bias: self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))

    def forward(self, edge_index, edge_type, query_type, entity_embed, rel_embed):
        device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_ent = entity_embed.size(0)
        self.out_index, self.out_type = edge_index, edge_type

        if self.p.loop:
            self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(device)
            self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(device)

        self.out_norm = self.compute_norm(self.out_index, num_ent, mode='out')  # 入度

        loop_res = torch.matmul(entity_embed, self.w_loop) if not self.p.loop else self.propagate(
            self.p.gcn_pool_fn, self.loop_index, edge_type=self.loop_type, query_type=query_type, 
            ent_embed=entity_embed, rel_embed=rel_embed, edge_norm=None, mode='loop')
        out_res = self.propagate(self.p.gcn_pool_fn, self.out_index, edge_type=self.out_type, query_type=query_type, 
                                 ent_embed=entity_embed, rel_embed=rel_embed, edge_norm=self.out_norm, mode='out')
        out = self.drop(out_res) * (1 / 2) + loop_res * (1 / 2)

        if self.p.bias: out = out + self.bias
        out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]  # Ignoring the self loop inserted

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.gcn_opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.gcn_opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.gcn_opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    def calculate_rel_attention(self, entity_embed, rel_embed, edge_index, edge_type):
        src_ent_embed = entity_embed[edge_index[0]]
        tgt_ent_embed = entity_embed[edge_index[1]]
        tmp_edge_type = edge_type - self.p.num_relations if edge_type[0] >= self.p.num_relations else edge_type + self.p.num_relations
        tmp_rel_embed = rel_embed[tmp_edge_type]
        target_embed = self.rel_transform(src_ent_embed, tmp_rel_embed)
        similarity = torch.mul(tgt_ent_embed, target_embed).sum(dim=1)  # torch.bmm(src_ent_embed.unsqueeze(dim=1), target_embed.unsqueeze(dim=1).transpose(2, 1)).squeeze(dim=1).squeeze(dim=1)
        attention_matrix = torch.full((self.p.num_entities, rel_embed.size(0)), -np.inf).float().to(self.p.device)
        attention_matrix[edge_index[0], edge_type] = similarity
        attention_matrix = torch.softmax(attention_matrix, dim=-1)
        return attention_matrix[edge_index[0], edge_type]

    def message(self, edge_index, edge_type, query_type, ent_embed, rel_embed, edge_norm, mode):
        ent_num = len(ent_embed)
        x_j = ent_embed[edge_index[1] % ent_num]
        weight = getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        if mode != 'loop':
            query_emb = torch.index_select(rel_embed, 0, query_type)
            alpha = torch.exp(self.act(self.a(torch.cat([self.W(rel_emb), self.W(query_emb)], dim=-1))))
            sum_alpha = scatter(alpha, index=edge_index[0], dim=0, reduce='add')
            alpha = alpha / sum_alpha[edge_index[0]]
        # attention = torch.full((len(ent_embed), len(ent_embed)), fill_value=-1e20).to(edge_index.device)
        # attention[edge_index[0], edge_index[1]] = alpha
        # attention = torch.softmax(attention, dim=-1)
        # dist = attention[edge_index[0], edge_index[1]]
        # print('xj_rel shape: ', xj_rel.shape)
        out = torch.mm(xj_rel, weight)  # W_\lambda(r) * \phi(e_u, e_r)
        if mode != 'loop':
            out = out * alpha
        # print('out shape: ', out.shape)
        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent, mode='out'):
        row, col = edge_index % num_ent
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges
        # deg = self.scatter_add(edge_weight, row, dim_size=num_ent)
        # deg_inv = deg.pow(-0.5)  # D^{-0.5}
        # deg_inv[deg_inv == float('inf')] = 0
        # norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}
        norm = deg[row] if mode == 'out' else deg[col]
        norm = norm.pow(-1)
        norm[norm == float('inf')] = 0
        return norm

    def scatter_add(self, src, index, dim_size):
        row_index = torch.zeros_like(index).long()
        out = torch.sparse_coo_tensor((row_index, index), src, size=(1, dim_size)).to_dense()[0]
        return out



class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh if params.gcn_aggr_act == 'tanh' else torch.relu
        print('using activate function {}'.format(self.act.__name__))
        self.loss = torch.nn.BCELoss()


class CompGCNBase(BaseModel):
    def __init__(self, in_dim, out_dim, params=None):
        super(CompGCNBase, self).__init__(params)

        self.gcn_embed_dim = out_dim
        self.gcn_dim = out_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        self.gcn_layer = self.p.gcn_layer
        self.init_dim = in_dim
        
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)

        self.conv1 = CompGCNConv(self.init_dim, self.gcn_dim, act=self.act, params=self.p)
        self.conv2 = CompGCNConv(self.gcn_dim, self.gcn_embed_dim, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

    def forward_base(self, edge_index, edge_type, query_type, ent_embeds, rel_embeds):
        x, r = self.conv1(edge_index, edge_type, query_type, ent_embeds, rel_embed=rel_embeds)
        x = self.hidden_drop(x)
        x, r = self.conv2(edge_index, edge_type, query_type, x, rel_embed=r) if self.gcn_layer == 2 else (x, r)
        x = self.feature_drop(x) if self.gcn_layer == 2 else x

        return x, r
