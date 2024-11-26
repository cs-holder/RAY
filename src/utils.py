import numpy as np
import scipy
import json, os
import networkx
import logging
import tqdm

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def load_index(path):
    name2ids = {}
    with open(path) as file:
        for i, line in enumerate(file):
            line = line.strip().split()
            if len(line) == 1:
                name2ids[line[0]] = i
            else:
                idx, name = line
                name2ids[name] = int(idx)
    return name2ids


def save_index(id2names, path):
    with open(path, "w") as output:
        for idx in range(len(id2names)):
            output.write(str(idx) + '\t' + id2names[idx] + "\n")


def load_facts(path):
    facts = []
    with open(path) as file:
        for line in file:
            facts.append(line.strip().split())
    return facts


def translate_facts(facts, entity_dict, relation_dict):
    facts = [(entity_dict[head], relation_dict[relation], entity_dict[tail]) for head, relation, tail in facts]
    return facts


def save_facts(facts, file_path):
    with open(file_path, "w") as output:
        for head, relation, tail in facts:
            output.write("{}\t{}\t{}\n".format(head, relation, tail))


def add_reverse_relations(facts):
    reverse_facts = []
    for head, relation, tail in facts:
        reverse_facts.append((tail, inverse_relation(relation), head))
    return reverse_facts


def build_dict(facts, entity_dict=None, relation_dict=None):
    if entity_dict is None:
        entity_dict = {}
    if relation_dict is None:
        relation_dict = {}
    for head, relation, tail in facts:
        if head not in entity_dict:
            entity_dict[head] = len(entity_dict)
        if tail not in entity_dict:
            entity_dict[tail] = len(entity_dict)
        if relation not in relation_dict:
            relation_dict[relation] = len(relation_dict)
    return entity_dict, relation_dict


class ListConcat:
    def __init__(self, *lists):
        self.lists = lists
        self.lengths = []
        length = 0
        for l in lists:
            length += len(l)
            self.lengths.append(length)

    def __len__(self):
        return self.lengths[-1]

    def __getitem__(self, item):
        last_length = 0
        for i, length in enumerate(self.lengths):
            if item < length:
                return self.lists[i][item - last_length]
            last_length = length


def serialize(obj, path, in_json=False):
    if isinstance(obj, np.ndarray):
        np.save(path, obj)
    elif isinstance(obj, scipy.sparse.csr.csr_matrix) or isinstance(obj, scipy.sparse.csc.csc_matrix):
        scipy.sparse.save_npz(path, obj)
    elif in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)


def unserialize(path, form=None):
    if form is None:
        form = os.path.basename(path).split(".")[-1]
    if form == "npy":
        return np.load(path)
    elif form == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)


def inverse_relation(relation: str):
    if relation.endswith('_inv'):
        return relation[:-4]
    else:
        return relation + '_inv'


def draw_graph(G: networkx.Graph):
    pos = networkx.spring_layout(G)
    networkx.draw_networkx(G, with_labels=True)
    networkx.draw_networkx_edge_labels(G, pos, edge_labels={(a, b): c for a, b, c in G.edges(keys=True)})


def read_hyper_result(result_path):
    best_valid_mrr = 0.0
    key = None
    current = {'valid': {}, 'test': {}}
    best = {'valid': {}, 'test': {}}
    is_best = False
    with open(result_path) as file:
        for line in file:
            line = line.strip()
            if line == 'Validation:':
                if is_best:
                    best[key].update(current[key])
                key = 'valid'
            elif line == 'Test:':
                if is_best:
                    best[key].update(current[key])
                key = 'test'
            elif key is not None and line.find(':') != -1 and not line.startswith('Relation'):
                name, value = line.split(':')
                current[key][name] = float(value)
            else:
                if key == 'test':
                    if current['valid']['Hits @10'] > best_valid_mrr:
                        is_best = True
                        best_valid_mrr = current['valid']['Hits @10']
                    else:
                        is_best = False
                    if is_best:
                        best['valid'].update(current['valid'])
                        best['test'].update(current['test'])
    return best



import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
from itertools import repeat
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2

    def rfft(x, d):
        t = rfft2(x, dim=(-d))
        return torch.stack((t.real, t.imag), -1)

    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=(-d))


def maybe_dim_size(index, dim_size=None):
    if dim_size is not None:
        return dim_size
    return index.max().item() + 1 if index.numel() > 0 else 0


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0.):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        if index.numel() > 0:
            index = index.view(index_size).expand_as(src)
        else:  # pragma: no cover
            # PyTorch has a bug when view is used on zero-element tensors.
            index = src.new_empty(index_size, dtype=torch.long)

    # Broadcasting capabilties: Expand dimensions to match.
    if src.dim() != index.dim():
        raise ValueError(
            ('Number of dimensions of src and index tensor do not match, '
             'got {} and {}').format(src.dim(), index.dim()))

    expand_size = []
    for s, i in zip(src.size(), index.size()):
        expand_size += [-1 if s == i and s != 1 and i != 1 else max(i, s)]
    src = src.expand(expand_size)
    index = index.expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        out_size = list(src.size())
        dim_size = maybe_dim_size(index, dim_size)
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim


def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0.):
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    return out.scatter_add_(dim, index, src)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None, fill_value=0.):
    out = scatter_add(src, index, dim, out, dim_size, fill_value)
    count = scatter_add(torch.ones_like(src), index, dim, None, out.size(dim))
    return out / count.clamp(min=1)


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    return irfft(com_mult(conj(rfft(a, 1)), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))