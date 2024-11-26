import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions
import torch_scatter
from torch_geometric.utils import degree
from collections import deque
import networkx
from networkx.algorithms.shortest_paths.generic import shortest_path_length
import math
import numpy as np
import io
import random
import pdb, argparse

import src.grapher
from src.torch_utils import list2tensor
from src.compgcn import CompGCNBase
# from src.rspmm import generalized_rspmm

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Summary(nn.Module):
    def __init__(self, hidden_size, graph: src.grapher.KG, entity_embeddings, relation_embeddings):
        nn.Module.__init__(self)
        self.graph = graph
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.embed_size = len(entity_embeddings)
        self.hidden_size = hidden_size
        self.neighbor_layer = nn.Linear(2 * self.embed_size, self.hidden_size)
        self.transform_layer = nn.Linear(self.embed_size, self.hidden_size)
        self.relation_layer = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.relation_activation = nn.ReLU()
        self.predict_layer = nn.Linear(self.hidden_size, len(relation_embeddings) - 1)
        self.node_activation = nn.ReLU()

    def forward(self, support_pairs, predict=False, evaluate=False):
        batch_size = len(support_pairs)
        # create the aggregate data
        entities, neighbor_entities, neighbor_relations, offsets = [], [], [], []
        for batch_id, pair in enumerate(support_pairs):
            for entity in pair:
                entities.append(entity)
                edges = list(self.graph.my_edges(entity, batch_id=batch_id if not evaluate else None))
                offsets.append(len(neighbor_entities))
                neighbor_entities.extend(map(lambda x: x[1], edges))
                neighbor_relations.extend(map(lambda x: x[2], edges))
        # transform to torch.Tensor
        entities = torch.tensor(entities, dtype=torch.long, device=self.entity_embeddings.device)
        neighbor_entities = torch.tensor(neighbor_entities, dtype=torch.long,
                                         device=self.entity_embeddings.device)
        neighbor_relations = torch.tensor(neighbor_relations, dtype=torch.long,
                                          device=self.relation_embeddings.device)
        offsets = torch.tensor(offsets, dtype=torch.long, device=self.entity_embeddings.device)
        # retrieve entity embeddings and transform
        entity_embeddings = self.entity_embeddings[entities]
        entity_embeddings = self.transform_layer(entity_embeddings)
        # retrieve neighbor embeddings and aggregate
        neighbor_entity_embeds = nn.functional.embedding_bag(neighbor_entities, self.entity_embeddings, offsets, sparse=False)
        neighbor_relation_embeds = nn.functional.embedding_bag(neighbor_relations, self.relation_embeddings, offsets, sparse=False)
        # concatenate aggregate results and transform
        neighbor_embeddings = torch.cat((neighbor_entity_embeds, neighbor_relation_embeds), dim=-1)
        neighbor_embeddings = self.neighbor_layer(neighbor_embeddings)
        node_embeddings = self.node_activation(entity_embeddings + neighbor_embeddings)
        node_embeddings = node_embeddings.view(batch_size, -1)
        pair_embeddings = self.relation_activation(self.relation_layer(node_embeddings))
        if predict:
            scores = self.predict_layer(pair_embeddings)
            return scores
        else:
            return pair_embeddings


class CogGraph:
    def __init__(self, graph: src.grapher.KG, entity_dict: dict, relation_dict: dict, entity_pad: int, relation_pad: int,
                 max_nodes: int, max_neighbors: int, topk: int, device, apply_global_info=False):
        self.graph = graph
        # the padding id for entities and relations
        self.entity_pad = entity_pad
        self.relation_pad = relation_pad
        self.node_pad = max_nodes
        self.node_pos_pad = max_nodes + 1
        self.entity_dict = entity_dict
        self.entity_num = len(entity_dict) + 1
        self.id2entity = sorted(self.entity_dict.keys(), key=self.entity_dict.get)
        self.relation_dict = relation_dict
        self.id2relation = sorted(self.relation_dict.keys(), key=self.relation_dict.get)
        self.self_loop_rel = self.relation_dict['SELF_LOOP']
        self.max_nodes = max_nodes                                  # max_nodes 应该是每次采样出子图的大小的上限？
        self.max_neighbors = max_neighbors
        self.device = device
        self.debug = False
        self.topk = topk
        self.apply_global_info = apply_global_info

    def init(self, start_entities: list, other_correct_answers, evaluate=False):
        self.evaluate = evaluate
        self.batch_size = len(start_entities)
        # self.other_correct_answers = list2tensor(other_correct_answers, padding_idx=self.entity_pad, dtype=torch.long, device=self.device)
        self.other_correct_answers = [np.array(list(answer_set)) for answer_set in other_correct_answers]
        batch_index = torch.arange(0, self.batch_size, dtype=torch.long, device=self.device)
        # each line is the head entity and relation type
        self.neighbor_matrix = torch.zeros(self.batch_size, self.max_nodes + 2, self.max_neighbors, 2, dtype=torch.long, device=self.device)
        # padding the neighbors
        self.neighbor_matrix[:, :, :, 0] = self.node_pad
        self.neighbor_matrix[:, :, :, 1] = self.relation_pad
        # neighbor number of each node
        self.neighbor_nums = torch.zeros(self.batch_size, self.max_nodes + 2, dtype=torch.long, device=self.device)
        self.stop_states = [False for _ in range(self.batch_size)]
        self.frontier_queues = [deque([start_entity]) for start_entity in start_entities]
        self.node_lists = [[start_entity] for start_entity in start_entities]
        # self.antecedents = [[set()] for _ in range(self.batch_size)]
        self.entity2node = [{start_entity: 0} for start_entity in start_entities]
        self.entity_translate = torch.full((self.batch_size, self.entity_num), fill_value=self.node_pad,
                                           dtype=torch.long, device=self.device)        # 放的应该是实体在子图里的索引
        self.entity_translate[batch_index, start_entities] = 0
        self.current_nodes = [{0} for _ in range(self.batch_size)]
        if self.debug:
            self.debug_outputs = [io.StringIO() for _ in range(self.batch_size)]

    def to_networkx(self):
        graphs = []
        for batch_id in range(self.batch_size):
            graph = networkx.MultiDiGraph()
            node_list = self.node_lists[batch_id]
            for node_id in range(len(self.node_lists[batch_id])):
                neighbor_num = self.neighbor_nums[batch_id, node_id]
                for neighbor_node, neighbor_relation in self.neighbor_matrix[batch_id, node_id, :neighbor_num].tolist():
                    graph.add_edge(self.id2entity[node_list[neighbor_node]], self.id2entity[node_list[node_id]],
                                   self.id2relation[neighbor_relation])
            graphs.append(graph)
        return graphs

    def step(self, current, last_step=False):
        """
        :return current: current entities (batch_size, rollout_num)
                         current nodes (batch_size, rollout_num)
                         current masks (batch_size, rollout_num)
                candidates: node id (batch_size, rollout_num, max_neighbors, )
                            entity id (batch_size, rollout_num, max_neighbors, )
                            relation id (batch_size, rollout_num, max_neighbors, )
                max_neighbors can be dynamic
        """
        current_entities, current_nodes, current_masks = current
        device = current_entities.device
        batch_index = torch.arange(0, self.batch_size, device=device)
        # current_nodes = self.entity_translate[batch_index.unsqueeze(-1), current_entities]
        # TODO change KG.quick_edges
        candidates, candidate_masks = self.graph.quick_edges(current_entities.cpu(), current_masks.cpu())
        candidates, candidate_masks = candidates.to(device), candidate_masks.to(device)
        candidate_entities, candidate_relations = candidates.select(-1, 0), candidates.select(-1, 1)
        candidate_masks &= current_masks.unsqueeze(dim=-1)
        
        if last_step:
            for batch_id in range(self.batch_size):
                other_masks = np.isin(candidate_entities[batch_id].cpu().numpy(), self.other_correct_answers[batch_id],
                                      invert=True)
                candidate_masks[batch_id] &= torch.from_numpy(other_masks).bool().to(candidate_masks.device)               # 这是禁止在训练当前样本时能够到达其他候选答案处
        
        batch_inds = batch_index.view(-1, 1, 1).expand_as(candidate_entities)
        batch_inds = torch.masked_select(batch_inds, candidate_masks)
        rollout_inds = torch.arange(current_entities.shape[1]).view(1, -1, 1).expand_as(candidate_entities).to(current_entities.device)
        rollout_inds = torch.masked_select(rollout_inds, candidate_masks)
        action_space_inds = torch.arange(candidate_entities.shape[-1]).view(1, 1, -1).expand_as(candidate_entities).to(current_entities.device)
        action_space_inds = torch.masked_select(action_space_inds, candidate_masks)
        flat_current_entities = current_entities.unsqueeze(dim=-1).expand_as(candidate_entities)
        flat_current_entities = torch.masked_select(flat_current_entities, candidate_masks)
        flat_candidate_entities = torch.masked_select(candidate_entities, candidate_masks)
        flat_candidate_relations = torch.masked_select(candidate_relations, candidate_masks)
        if self.apply_global_info:
            flat_current_nodes = current_nodes.unsqueeze(dim=-1).expand_as(candidate_entities)
            flat_current_nodes = torch.masked_select(flat_current_nodes, candidate_masks)
            edges = torch.stack([batch_inds, rollout_inds, action_space_inds, flat_current_entities, 
                                flat_candidate_relations, flat_candidate_entities, flat_current_nodes], dim=-1)
        else:
            edges = torch.stack([batch_inds, rollout_inds, action_space_inds, flat_current_entities, 
                                flat_candidate_relations, flat_candidate_entities], dim=-1)
        
        nodes, edges, old_nodes_new_idx, old_nodes_old_idx = self.compress(edges)
        return (candidate_entities, candidate_relations, candidate_masks), nodes, edges, old_nodes_new_idx, old_nodes_old_idx

    def compress(self, edges: torch.LongTensor, old_head_index: torch.LongTensor = None):
        # index to nodes
        head_nodes, head_index = torch.unique(edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(edges[:,[0,5]], dim=0, sorted=True, return_inverse=True)
        # edges = torch.cat([edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        edges = torch.cat([edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
        mask = (edges[:,4] == self.self_loop_rel)                 # 自循环边
        # loop_head_index_list = head_index[mask].cpu().tolist()
        # loop_head_index = sorted(set(loop_head_index_list), key=loop_head_index_list.index)
        # old_idx = np.argsort(loop_head_index)
        old_nodes_old_idx, old_idx = head_index[mask].sort()
        # old_nodes_old_idx = head_index[mask][old_idx]
        old_nodes_new_idx = tail_index[mask][old_idx]
        if old_head_index is not None:
            old_head_index = old_head_index[mask][old_idx]
            return tail_nodes, edges, old_nodes_new_idx, old_nodes_old_idx, old_head_index
        return tail_nodes, edges, old_nodes_new_idx, old_nodes_old_idx


class Agent(nn.Module):
    def __init__(self, entity_embeddings, relation_embeddings, max_nodes: int, agent_num: int, embed_size: int, 
                 hidden_size: int, state_size: int, query_size: int, message: bool, entity_embed: bool, dropout: float, 
                 message_func="rnn", eps=1e-6, activation='relu', gcn_params=None, apply_global_info=False):
        nn.Module.__init__(self)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.sqrt_embed_size = math.sqrt(self.embed_size)
        self.max_nodes = max_nodes
        self.agent_num = agent_num
        self.apply_global_info = apply_global_info
        if query_size is None:
            query_size = hidden_size
        self.query_size = query_size
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.use_message = message
        self.use_entity_embed = entity_embed
        input_size = 2 * self.embed_size if self.use_message else self.embed_size
        self.hiddenRNN = nn.GRUCell(input_size=input_size, hidden_size=self.hidden_size)
        self.update_layer = nn.Linear(input_size, self.hidden_size)
        self.update_activation = nn.LeakyReLU()
        nexthop_input_size = self.hidden_size + self.query_size + self.embed_size + self.state_size if self.apply_global_info \
            else self.hidden_size + self.query_size + self.embed_size
        self.nexthop_layer = nn.Linear(nexthop_input_size, self.hidden_size * self.agent_num)
        self.nexthop_activation = nn.LeakyReLU()
        self.candidate_layer = nn.Linear(2 * self.embed_size + self.hidden_size, self.hidden_size * self.agent_num)
        self.candidate_activation = nn.LeakyReLU()
        self.rank_layer = nn.Linear(self.hidden_size + self.query_size, 1)
        # this should combine with the loss function
        self.rank_activation = nn.Sequential()
        
        self.agent_flag = nn.Parameter(torch.rand(self.hidden_size, self.agent_num))
        self.agent_allocate_layer = nn.Linear(self.embed_size + self.query_size + self.hidden_size, self.hidden_size)
        self.agent_allocate_activation = nn.LeakyReLU()
        
        self.dropout = nn.Dropout(dropout)
        self.W_final = nn.Linear(self.hidden_size, 1, bias=False)         # get score
        self.gate = nn.GRU(self.hidden_size, self.hidden_size)
        
        if self.apply_global_info:
            self.sub_graph = None
            self.sg_node_embeddings = None
            self.sg_rel_embeddings = None
            self.edge_gate = nn.GRU(self.embed_size, self.embed_size)
            self.gcn_embeder = CompGCNBase(self.embed_size, self.state_size, argparse.Namespace(**gcn_params))
            self.node2embed_layer = nn.Linear(self.hidden_size, self.embed_size)
            self.node2embed_activation = nn.LeakyReLU()
        
        self.Ws_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wr_attn = nn.Linear(self.embed_size, self.hidden_size, bias=False)
        self.Wqr_attn = nn.Linear(self.embed_size, self.hidden_size)
        self.w_alpha  = nn.Linear(self.hidden_size, 1)
        self.We2h = nn.Linear(self.embed_size, self.hidden_size)

        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act = nn.ReLU()
        self.debug = False
        
        self.message_func = message_func
        self.eps = eps
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        self.update_linear = nn.Linear(13 * hidden_size, hidden_size)
        
        print('Initializing modules...')
        self.initialize_modules()

    def initialize_modules(self):
        nn.init.xavier_uniform_(self.update_layer.weight)
        nn.init.xavier_uniform_(self.nexthop_layer.weight)
        nn.init.xavier_uniform_(self.candidate_layer.weight)
        
        nn.init.xavier_uniform_(self.agent_flag.data)
        nn.init.xavier_uniform_(self.agent_allocate_layer.weight)
        nn.init.xavier_uniform_(self.update_linear.weight)
        
        nn.init.xavier_uniform_(self.W_final.weight)
        nn.init.xavier_uniform_(self.Ws_attn.weight)
        nn.init.xavier_uniform_(self.Wr_attn.weight)
        nn.init.xavier_uniform_(self.Wqr_attn.weight)
        nn.init.xavier_uniform_(self.w_alpha.weight)
        nn.init.xavier_uniform_(self.We2h.weight)
        
        if self.apply_global_info:
            nn.init.xavier_uniform_(self.node2embed_layer.weight)
            for name, param in self.edge_gate.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
        
        for module in [self.gate, self.hiddenRNN]:
            for name, param in module.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def debug_tensor(self, t: torch.Tensor, local_info):
        # print a tensor according to its batch_id
        if self.debug:
            for batch_id in range(self.batch_size):
                self.debug_outputs[batch_id].write(str(t))

    def init(self, start_entities: torch.Tensor, query_representations=None):
        self.batch_size = start_entities.size(0)
        self.node_embeddings = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float,
                                           device=self.entity_embeddings.device)
        if query_representations is not None:
            if query_representations.size(0) == 1:
                query_representations = query_representations.expand(self.batch_size, -1)
            self.query_representations = query_representations
        else:
            self.query_representations = torch.zeros(self.batch_size, self.embed_size, dtype=torch.float,
                                                     device=self.entity_embeddings.device)
        init_embeddings = query_representations #torch.zeros(self.batch_size, self.hidden_size, device=start_entities.device)
        if self.use_message:
            entity_embeddings = self.entity_embeddings[start_entities]
            init_embeddings = torch.cat((entity_embeddings, init_embeddings), dim=-1)
        self.node_embeddings = self.update_activation(self.update_layer(init_embeddings))
        self.h0 = torch.zeros((1, self.batch_size, self.hidden_size)).to(start_entities.device)
        if self.apply_global_info:
            self.sg_node_embeddings = self.node2embed_activation(self.node2embed_layer(self.node_embeddings))
            self.edge_embeddings = None
            self.rel_h0 = torch.zeros((1, self.batch_size, self.embed_size)).to(start_entities.device)
            self.global_embeds = query_representations
            self.sub_graph = None
            self.sg_rel_embeddings = None
        if self.debug:
            self.debug_outputs = [io.StringIO() for _ in range(self.batch_size)]

    def update_edge_embeds(self, nodes, sampled_head_index, edges):
        edge_rels, tail_index = edges[:, 4], edges[:, -1]
        rel_h0 = self.rel_h0[:, sampled_head_index]
        rel_embeds = self.relation_embeddings[edge_rels]
        new_edge_embeds, h0 = self.edge_gate(rel_embeds.unsqueeze(dim=0), rel_h0)
        self.rel_h0 = torch_scatter.scatter(h0.squeeze(dim=0), tail_index, dim=0, dim_size=nodes.size(0)).unsqueeze(dim=0)
        self.edge_embeddings = new_edge_embeds.squeeze(dim=0)

    def update_graph(self, nodes, edges, query_relations):
        new_tails = edges[:, -1] + self.sg_node_embeddings.shape[0]
        if self.sub_graph is None:
            # self.sub_graph = torch.stack([edges[:, 0], new_rels, new_tails], dim=-1)
            self.sub_graph = torch.stack([edges[:, 6], edges[:, 4], new_tails], dim=-1)
            self.query_relations = query_relations
        else:
            new_rels = torch.arange(self.edge_embeddings.shape[0]).to(nodes.device) + len(self.relation_embeddings)
            if self.sg_rel_embeddings is not None:
                new_rels += self.sg_rel_embeddings.shape[0]
                self.sg_rel_embeddings = torch.cat([self.sg_rel_embeddings, self.edge_embeddings], dim=0)
            else:
                self.sg_rel_embeddings = self.edge_embeddings
            # self.sub_graph = torch.cat([self.sub_graph, torch.stack([edges[:, 6], edges[:, 4], new_tails], dim=-1), 
            #                             torch.stack([edges[:, 0], new_rels, new_tails], dim=-1)], dim=0)
            # self.query_relations = torch.cat([self.query_relations, query_relations, query_relations], dim=0)
            self.sub_graph = torch.cat([self.sub_graph, torch.stack([edges[:, 0], new_rels, new_tails], dim=-1)], dim=0)
            self.query_relations = torch.cat([self.query_relations, query_relations], dim=0)
        self.sg_node_embeddings = torch.cat([self.sg_node_embeddings, self.node2embed_activation(self.node2embed_layer(self.node_embeddings))], dim=0)
        
    def update_global_graph_states(self, start_entities):
        edge_index, edge_type = self.sub_graph[:, [0, 2]].t(), self.sub_graph[:, 1]
        if self.sg_rel_embeddings is not None:
            rel_embeds = torch.cat([self.relation_embeddings, self.sg_rel_embeddings], dim=0)
        else:
            rel_embeds = self.relation_embeddings
        ent_embeds, _ = self.gcn_embeder.forward_base(edge_index, edge_type, self.query_relations, self.sg_node_embeddings, rel_embeds)
        self.global_embeds = ent_embeds[start_entities]
    
    def aggregate_cokgr(self, nodes, edges, alpha, sampled_head_index, old_nodes_new_idx, old_head_index):
        """
        :param aim_data: (batch_size, topk) ids of updated nodes, used to update hidden representations
                         (batch_size, topk) ids of updated entities
                         (batch_size, ) number of aims
        :param neighbor_data: (batch_size, topk, max_neighbors, 2) node and relation type
                              (batch_size, topk) number of neighbors
        :return: None
        """     
        # batch_inds, current_entities, current_nodes = edges[:, 0], edges[:, 1], edges[:, 4]
        candidate_relations, candidate_entities, candidate_nodes = edges[:, 4], edges[:, 5], edges[:, -1]
        # (batch_size, topk, embed_size) get the entity embeddings of aims to update
        entity_embeddings = self.entity_embeddings[candidate_entities]
        relation_embeddings = self.relation_embeddings[candidate_relations]
        # (batch_size, topk, max_neighbors, embed_size)
        # get the hidden representations and relation embeddings of neighbors
        node_embeddings = self.node_embeddings[sampled_head_index]
        if self.use_message:
            # (batch_size, topk, max_neighbors, 2 * embed_size) concatenated neighbor embeddings
            neighbor_embeddings = torch.cat((entity_embeddings, relation_embeddings), dim=-1)
        else:
            # (batch_size, topk, max_neighbors, embed_size + hidden_size) concatenated neighbor embeddings
            neighbor_embeddings = relation_embeddings
        # neighbor_embeddings = self.update_activation(self.update_layer(neighbor_embeddings))
        if self.message_func == 'rnn':
            updated_embeddings = self.hiddenRNN(neighbor_embeddings, node_embeddings)
        elif self.message_func == 'distmult':
            updated_embeddings = neighbor_embeddings * node_embeddings
        elif self.message_func == 'transe':
            updated_embeddings = neighbor_embeddings + node_embeddings
            
        if candidate_nodes.max() >= nodes.shape[0]:
            print('error')
        
        sum_feature = torch_scatter.scatter(updated_embeddings * alpha, index=candidate_nodes, dim=0, dim_size=nodes.size(0), reduce='sum')
        sq_sum_feature = torch_scatter.scatter(updated_embeddings ** 2 * alpha, index=candidate_nodes, dim=0, dim_size=nodes.size(0), reduce='sum')
        max_feature = torch_scatter.scatter(updated_embeddings * alpha, index=candidate_nodes, dim=0, dim_size=nodes.size(0), reduce="max")
        min_feature = torch_scatter.scatter(updated_embeddings * alpha, index=candidate_nodes, dim=0, dim_size=nodes.size(0), reduce="min")
        degree_out = degree(candidate_nodes, nodes.size(0)).unsqueeze(-1) + 1
        if len(old_head_index):
            boundary = torch.zeros(nodes.size(0), updated_embeddings.size(1)).to(edges.device).index_copy_(
                0, old_nodes_new_idx, self.node_embeddings[old_head_index])
        else:
            boundary = torch.zeros(nodes.size(0), updated_embeddings.size(1)).to(edges.device)
        sum_feature = (sum_feature + boundary) / degree_out
        sq_sum_feature = (sq_sum_feature + boundary ** 2) / degree_out
        max_feature = torch.max(max_feature, boundary)
        min_feature = torch.min(min_feature, boundary) # (node, batch_size * input_dim)
        std_feature = (sq_sum_feature - sum_feature ** 2).clamp(min=self.eps).sqrt()
        features = torch.cat([sum_feature.unsqueeze(-1), max_feature.unsqueeze(-1), min_feature.unsqueeze(-1), std_feature.unsqueeze(-1)], dim=-1)
        features = features.flatten(-2)
        scale = degree_out.log()
        scale = scale / scale.mean()
        scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
        update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        
        output = self.update_linear(torch.cat([boundary, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        self.node_embeddings = output
          
    def aggregate_redgnn(self, candidate_nodes, edges, old_nodes_new_idx, old_head_index):
        hidden_new = self.node_embeddings
        if len(old_head_index):
            h0 = torch.zeros(1, candidate_nodes.size(0), hidden_new.size(1)).to(edges.device).index_copy_(
                1, old_nodes_new_idx, self.h0[:, old_head_index])
        else:
            h0 = torch.zeros(1, candidate_nodes.size(0), hidden_new.size(1)).to(edges.device)
        hidden = self.dropout(hidden_new)
        hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
        self.node_embeddings = hidden.squeeze(0)
        self.h0 = h0
    
    def aggregate(self, nodes, edges, alpha, sampled_head_index, old_nodes_new_idx, old_head_index, apply_red_aggr=False):
        self.aggregate_cokgr(nodes, edges, alpha, sampled_head_index, old_nodes_new_idx, old_head_index)
        if apply_red_aggr:
            self.aggregate_redgnn(nodes, edges, old_nodes_new_idx, old_head_index)
    
    def next_hop_cogkr(self, rollout_size, action_space_size, candidate_nodes, edges, old_nodes_new_idx, old_nodes_old_idx):
        """
        :param currents: (batch_size, rollout_num) pos of current entities
        :param candidates: entity id (batch_size, rollout_num, max_neighbors)
                           node pos (batch_size, rollout_num, max_neighbors)
                           relation id (batch_size, rollout_num, max_neighbors)
                           mask (batch_size, rollout_num, max_neighbors)
        :param topk: topk actions to select
        :return: entity id (batch_size, topk), relation id (batch_size, topk) mask (batch_size, topk)
        """
        current_head_nodes, current_nodes, current_entities = edges[:, 0], edges[:, -2], edges[:, 3]
        candidate_nodes, candidate_entities, candidate_relations = edges[:, -1], edges[:, 5], edges[:, 4]
        # (batch_size, embed_size) get the hidden representations of current nodes
        current_representations = self.node_embeddings[current_nodes]
        if self.use_entity_embed:
            current_embeddings = self.entity_embeddings[current_entities]
        else:
            current_embeddings = current_representations.new_zeros((current_entities.shape[0], self.embed_size))
        # concatenate the hidden states with query embeddings
        query_representations = self.query_representations[edges[:, 0]]
        if self.apply_global_info:
            global_embeddings = self.global_embeds[current_head_nodes]
            current_embeddings = torch.cat(
                (current_representations, query_representations, current_embeddings, global_embeddings), dim=-1)
        else:
            current_embeddings = torch.cat((current_representations, query_representations, current_embeddings), dim=-1)
        current_state = self.nexthop_activation(self.nexthop_layer(current_embeddings))
        # (batch_size, rollout_num, max_neighbors, hidden_size) get the node representations of candidates
        new_node_embeddings = torch.zeros(candidate_nodes.size(0), self.hidden_size).to(candidate_nodes.device).index_copy_(
            0, old_nodes_new_idx, self.node_embeddings[old_nodes_old_idx])
        # new_node_embeddings[old_nodes_new_idx] = self.node_embeddings
        node_embeddings = new_node_embeddings[candidate_nodes]
        # (batch_size, rollout_num, max_neighbors, embed_size) get the entity embeddings of candidates
        if self.use_entity_embed:
            entity_embeddings = self.entity_embeddings[candidate_entities]
        else:
            entity_embeddings = node_embeddings.new_zeros((candidate_entities.shape[0], self.embed_size))
        # (batch_size, rollout_num, max_neighbors, embed_size) get the relation embeddings of candidates
        relation_embeddings = self.relation_embeddings[candidate_relations]
        # (batch_size, max_neighbors, 2 * embed_size + hidden_size) concatenated representations
        candidate_embeddings = torch.cat((node_embeddings, relation_embeddings, entity_embeddings), dim=-1)
        # (batch_size, max_neighbors, embed_size) transformed representations
        candidate_embeddings = self.candidate_activation(self.candidate_layer(candidate_embeddings))
        # (batch_size, rollout_num, max_neighbors) (batch_size, rollout_num, hidden_size) * (batch_size, rollout_num, max_neighbors, hidden_size)
        # candidate_scores = (candidate_embeddings * current_state).sum(dim=-1)
        candidate_scores = (candidate_embeddings * current_state).reshape(edges.size(0), self.agent_num, -1).sum(dim=-1)         # |edge_size| * agent_num
        candidate_scores /= self.sqrt_embed_size
        
        if self.agent_num > 1:
            allocate_embedinngs = self.agent_allocate_activation(self.agent_allocate_layer(torch.cat([
                relation_embeddings, query_representations, current_representations], dim=-1)))
            allocate_dists = torch.softmax(torch.matmul(allocate_embedinngs, self.agent_flag), dim=-1)                     # |edge_size| * agent_num
            allocate_dist_entropy = -(allocate_dists * torch.log(allocate_dists + 1e-10)).sum(dim=-1).mean()
            candidate_scores = (candidate_scores * allocate_dists).sum(dim=-1)
        else:
            allocate_dist_entropy = 0.
            candidate_scores = candidate_scores.sum(dim=-1)
        
        attention = torch.full((self.batch_size, rollout_size, action_space_size), fill_value=-1e20).to(edges.device)
        attention[edges[:, 0], edges[:, 1], edges[:, 2]] = candidate_scores
        edges_index = torch.full_like(attention, fill_value=edges.size(0), dtype=torch.long)
        edges_index[edges[:, 0], edges[:, 1], edges[:, 2]] = torch.arange(edges.size(0)).to(edges.device)
        return attention, edges_index, allocate_dist_entropy

    def next_hop_redgnn(self, q_rel, edges):
        sub = edges[:,-2]
        rel = edges[:,4]

        hs = self.node_embeddings[sub]
        hr = self.relation_embeddings[rel]

        r_idx = edges[:,0]
        h_qr = self.relation_embeddings[q_rel][r_idx]

        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        return alpha
    
    def compute_score(self, batch_size, candidate_nodes, normalize=True):
        scores = self.W_final(self.node_embeddings).squeeze(-1)
        if normalize:
            scores_all = torch.ones((batch_size, len(self.entity_embeddings))).to(candidate_nodes.device) * (-1e20)         # non_visited entities have 0 scores
        else:
            scores_all = torch.zeros((batch_size, len(self.entity_embeddings))).to(candidate_nodes.device)         # non_visited entities have 0 scores  
        scores_all[[candidate_nodes[:,0], candidate_nodes[:,1]]] = scores
        if normalize:
            scores_all = torch.softmax(scores_all, dim=-1)
        
        # scores = self.W_final(self.node_embeddings).squeeze(-1)
        # scores_all = torch.zeros((batch_size, self.entity_embeddings.num_embeddings - 1)).cuda()         # non_visited entities have 0 scores
        # scores_all[candidate_nodes[:,0], candidate_nodes[:,1]] = scores
        return scores_all


class CogKR(nn.Module):
    def __init__(self, graph: src.grapher.KG, entity_dict: dict, relation_dict: dict, max_steps: int, max_nodes: int,
                 agent_num: int, max_neighbors: int,
                 embed_size: int, topk: list, device, hidden_size: int = None, state_size: int = None, reward_policy='direct', 
                 use_summary=True, baseline_lambda=0.0, onlyS=False, update_hidden=True,
                 message=True, entity_embed=True, sparse_embed=False, id2entity=None, id2relation=None, dropout=None, 
                 fuse_dist="mult", eval_mode="mult", message_func="distmult", eps=1e-6, apply_red_aggr=False, apply_final_red_aggr=False, 
                 apply_edge_weight=True, gcn_params=None, apply_global_info=False):
        nn.Module.__init__(self)
        self.graph = graph
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.entity_pad = len(entity_dict)
        self.relation_pad = len(relation_dict)
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        self.agent_num = agent_num
        self.max_neighbors = max_neighbors
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.onlyS = onlyS
        self.update_hidden = update_hidden
        if hidden_size is None:
            self.hidden_size = embed_size
        self.topk = topk
        self.entity_num = len(entity_dict) + 1
        # self.entity_embeddings = nn.Embedding(self.entity_num, embed_size, padding_idx=len(entity_dict),
        #                                       sparse=sparse_embed)
        self.entity_embeddings = torch.nn.Parameter(torch.zeros(self.entity_num, embed_size))
        nn.init.normal_(self.entity_embeddings)
        self.relation_num = len(relation_dict) + 1
        # self.relation_embeddings = nn.Embedding(self.relation_num, embed_size, padding_idx=len(relation_dict),
        #                                         sparse=sparse_embed)
        self.relation_embeddings = torch.nn.Parameter(torch.zeros(self.relation_num, embed_size))
        nn.init.normal_(self.relation_embeddings)
        # torch.nn.init.constant_(self.entity_embeddings.weight, 1.0)
        # torch.nn.init.constant_(self.relation_embeddings.weight, 1.0)
        self.cog_graph = CogGraph(self.graph, self.entity_dict, self.relation_dict, len(self.entity_dict), len(self.relation_dict), 
                                  self.max_nodes, self.max_neighbors, self.topk, device, apply_global_info=apply_global_info)
        # self.summary = Summary(self.hidden_size, graph, self.entity_embeddings, self.relation_embeddings)
        if use_summary:
            query_size = hidden_size
        else:
            query_size = embed_size
            print("Not use summary module")
        self.agent = Agent(self.entity_embeddings, self.relation_embeddings, self.max_nodes, self.agent_num, self.embed_size,
                           self.hidden_size, state_size, query_size=query_size, message=message, entity_embed=entity_embed, dropout=dropout, 
                           message_func=message_func, eps=eps, gcn_params=gcn_params, apply_global_info=apply_global_info)
        if self.onlyS:
            self.loss = nn.MarginRankingLoss(margin=1.0)
        else:
            self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.reward_policy = reward_policy
        self.reward_baseline = 0.0
        self.baseline_lambda = baseline_lambda
        
        self.fuse_dist = fuse_dist
        self.eval_mode = eval_mode
        self.apply_red_aggr = apply_red_aggr
        self.apply_final_red_aggr = apply_final_red_aggr
        self.apply_edge_weight = apply_edge_weight
        self.mlp = nn.Sequential()
        mlp = []
        # for i in range(self.max_steps - 1):
        mlp.append(nn.Linear(self.hidden_size + self.embed_size, self.hidden_size + self.embed_size))
        mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.hidden_size + self.embed_size, 1))
        self.mlp = nn.Sequential(*mlp)

    def find_correct_tails(self, node_lists, end_entities, only_last=False):
        assert len(node_lists) == len(end_entities)
        correct_batch, correct_nodes = [], []
        for batch_id in range(len(node_lists)):
            end_entity = end_entities[batch_id]
            found_correct = False
            if only_last:
                node_list = self.cog_graph.current_nodes[batch_id]
            else:
                node_list = range(0, len(self.cog_graph.node_lists[batch_id]))
            for node_id in node_list:
                if self.cog_graph.node_lists[batch_id][node_id] == end_entity:
                    correct_nodes.append(node_id)
                    found_correct = True
                    break
            if found_correct:
                correct_batch.append(batch_id)
        return correct_batch, correct_nodes

    def get_correct_path(self, batch_id, correct_tails, return_graph=False):
        correct_batch, correct_nodes = self.find_correct_tails([self.cog_graph.node_lists[batch_id]], correct_tails)
        graphs = self.cog_graph.to_networkx()
        if return_graph:
            reason_list = [{} for _ in range(len(correct_tails))]
        else:
            reason_list = [[] for _ in range(len(correct_tails))]
        for batch_id, node_id in zip(correct_batch, correct_nodes):
            correct_tail = self.id2entity[self.cog_graph.node_lists[batch_id][node_id]]
            head = self.id2entity[self.cog_graph.node_lists[batch_id][0]]
            graph = graphs[batch_id]
            if return_graph:
                nodes = shortest_path_length(graph, target=correct_tail)
                neighbor_dict = {}
                for node in nodes:
                    neighbor_dict[node] = []
                    for e1, e2, r in graph.edges(node, keys=True):
                        if e2 in nodes:
                            neighbor_dict[node].append((e1, e2, r))
                reason_list[batch_id] = neighbor_dict
            else:
                paths = list(networkx.algorithms.all_simple_paths(graphs[batch_id], head, correct_tail))
                reason_paths = []
                for path in paths:
                    reason_path = [path[0]]
                    last_node = path[0]
                    for node in path[1:]:
                        relation = list(
                            map(lambda x: x[2], filter(lambda x: x[1] == node, graph.edges(last_node, keys=True))))
                        last_node = node
                        reason_path.append((node, relation))
                    reason_paths.append(reason_path)
                reason_list[batch_id] = reason_paths
        return reason_list

    def loss_fn(self, attention, batch_size, batch_index, end_entities):
        correct_batch = attention[batch_index, end_entities] > 1e-10
        wrong_batch = batch_index[~correct_batch]
        correct_batch = batch_index[correct_batch]
        loss = -torch.log(attention[correct_batch, end_entities[correct_batch]] + 1e-10).sum()
        loss = loss - torch.log(1.01 - attention[wrong_batch].sum(dim=-1)).sum()
        # loss = loss / batch_size
        return loss
    
    def compute_score(self, batch_size, candidate_nodes, query_embeddings, normalize=True):
        logits = torch.cat([self.agent.node_embeddings, query_embeddings], dim=-1)
        scores = self.mlp(logits).squeeze(-1)
        if normalize:
            scores_all = torch.ones((batch_size, len(self.entity_embeddings))).to(candidate_nodes.device) * (-1e20)         # non_visited entities have 0 scores
        else:
            scores_all = torch.zeros((batch_size, len(self.entity_embeddings))).to(candidate_nodes.device)         # non_visited entities have 0 scores  
        scores_all[[candidate_nodes[:,0], candidate_nodes[:,1]]] = scores
        if normalize:
            scores_all = torch.softmax(scores_all, dim=-1)
        
        # scores = self.W_final(self.node_embeddings).squeeze(-1)
        # scores_all = torch.zeros((batch_size, self.entity_embeddings.num_embeddings - 1)).cuda()         # non_visited entities have 0 scores
        # scores_all[candidate_nodes[:,0], candidate_nodes[:,1]] = scores
        return scores_all
    
    def forward(self, start_entities: list, other_correct_answers: list, end_entities=None,
                support_pairs=None, relations=None, evaluate=False, candidates=None, do_sample=False, eval_mode='mult'):
        batch_size = len(start_entities)
        device = self.entity_embeddings.device
        batch_index = torch.arange(0, batch_size, device=device)
        # if support_pairs is not None:
        #     # support for evaluate
        #     support_embeddings = self.summary(support_pairs, evaluate=evaluate)
        # else:
        relations = torch.tensor(relations, device=device, dtype=torch.long)
        support_embeddings = self.relation_embeddings[relations]        
        self.cog_graph.init(start_entities, other_correct_answers, evaluate=evaluate)
        start_entities = torch.tensor(start_entities, device=device, dtype=torch.long)
        if end_entities is not None:
            end_entities = torch.tensor(end_entities, device=device, dtype=torch.long)
        self.agent.init(start_entities, query_representations=support_embeddings)
        # TODO: Normalize graph loss and entropy loss with time step
        if self.reward_policy == 'direct':
            attention = torch.zeros(batch_size, self.entity_num, device=device)
            attention[batch_index, start_entities] = 1
            entropy_loss, agent_entropy_loss = 0.0, 0.0
        else:
            graph_loss, entropy_loss = 0.0, 0.0
        current_entities = start_entities.unsqueeze(1)
        current_nodes = batch_index.unsqueeze(1)
        current_masks = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        for step in range(self.max_steps):
            candidates, candidate_nodes, edges, old_nodes_new_idx, old_nodes_old_idx = self.cog_graph.step((current_entities, current_nodes, current_masks), 
                                                                                                           step == self.max_steps - 1)
            candidate_entities, candidate_relations, candidate_masks = candidates
            rollout_size, action_space_size = candidate_entities.shape[1], candidate_entities.shape[2]
            final_scores, edges_index, allocate_dist_entropy = self.agent.next_hop_cogkr(rollout_size, action_space_size, candidate_nodes, 
                                                                                         edges, old_nodes_new_idx, old_nodes_old_idx)
            agent_entropy_loss += allocate_dist_entropy
            
            if self.reward_policy == 'direct':
                attention_scores = attention[batch_index.unsqueeze(1), current_entities]
                attention_scores = attention_scores.unsqueeze(dim=-1).expand_as(final_scores).reshape((batch_size, -1))
                final_scores = torch.softmax(final_scores.reshape(batch_size, -1), dim=-1)
                final_scores = final_scores * attention_scores
                entropy = -(final_scores * torch.log(final_scores + 1e-10)).sum(dim=-1).mean()
                entropy_loss += entropy
                if step != self.max_steps - 1:
                    if do_sample and not evaluate:
                        action_inds = torch.multinomial(final_scores, num_samples=min(self.topk[step], final_scores.size(-1)), replacement=False)
                        action_scores = torch.gather(final_scores, dim=1, index=action_inds)
                    else:
                        action_scores, action_inds = final_scores.topk(k=min(self.topk[step], final_scores.size(-1)), dim=-1, sorted=False)
                    
                    action_masks = candidate_masks.reshape((batch_size, -1))[batch_index.unsqueeze(-1), action_inds]
                    action_entities = candidate_entities.reshape((batch_size, -1))[batch_index.unsqueeze(-1), action_inds]
                    sample_edges_index = edges_index.reshape((batch_size, -1))[batch_index.unsqueeze(-1), action_inds]
                    flat_sample_edges_index = sample_edges_index[action_masks]
                    sorted_flat_sample_edges_index, _ = torch.sort(flat_sample_edges_index)
                    
                    sampled_edges = edges[sorted_flat_sample_edges_index]
                    sampled_head_index = sampled_edges[:, -2]
                    
                    candidate_nodes, edges, old_nodes_new_idx, _, old_head_index = self.cog_graph.compress(sampled_edges[:, :-2], 
                                                                                                           sampled_head_index)
                    
                    if self.apply_edge_weight:
                        alpha = self.agent.next_hop_redgnn(relations, edges)
                    else:
                        alpha = torch.ones((edges.size(0), 1), device=device)
                    
                    attention = torch_scatter.scatter_add(action_scores, action_entities, dim=-1, dim_size=self.entity_num)
                    # attention /= attention.sum(dim=-1, keepdim=True)
                    attention = torch.nn.functional.normalize(attention, p=1, dim=-1)
                    
                    if self.update_hidden:
                        self.agent.aggregate(candidate_nodes, edges, alpha, sampled_head_index, old_nodes_new_idx, old_head_index, self.apply_red_aggr)
                    
                    current_entities = torch.full((batch_size, candidate_nodes.size(0)), fill_value=self.entity_num - 1, dtype=torch.long, device=device)
                    current_masks = torch.full_like(current_entities, fill_value=False, dtype=torch.bool, device=device)
                    current_nodes = torch.full((batch_size, candidate_nodes.size(0)), fill_value=self.max_nodes, dtype=torch.long, device=device)
                    current_entities[edges[:, 0], edges[:, -1]] = edges[:, 5]
                    current_masks[edges[:, 0], edges[:, -1]] = True
                    # current_nodes[edges[:, 0], edges[:, -1]] = edges[:, 7] + self.agent.sg_node_embeddings.shape[0]
                    current_masks, inv_offset = torch.sort(current_masks.float(), dim=-1, descending=True)
                    current_entities = torch.gather(current_entities, dim=1, index=inv_offset)
                    max_current_num = torch.sum(current_masks, dim=-1).max().long().item()
                    # current_nodes = torch.gather(current_nodes, dim=1, index=inv_offset)
                    current_entities = current_entities[:, :max_current_num]
                    current_masks = current_masks[:, :max_current_num].bool()
                    # current_nodes = current_nodes[:, :max_current_num]
                    
                    if self.agent.apply_global_info:
                        current_nodes[edges[:, 0], edges[:, -1]] = edges[:, -1] + self.agent.sg_node_embeddings.shape[0]
                        current_nodes = torch.gather(current_nodes, dim=1, index=inv_offset)
                        current_nodes = current_nodes[:, :max_current_num]
                        self.agent.update_edge_embeds(candidate_nodes, sampled_head_index, edges)
                        self.agent.update_graph(candidate_nodes, edges, relations[sampled_edges[:, 0]])
                        self.agent.update_global_graph_states(batch_index)
                else:
                    if self.apply_edge_weight:
                        alpha = self.agent.next_hop_redgnn(relations, edges)
                    else:
                        alpha = torch.ones((edges.size(0), 1), device=device)
                    attention = torch_scatter.scatter_add(final_scores, candidate_entities.reshape((batch_size, -1)),
                                                          dim=-1, dim_size=self.entity_num)
                    attention = torch.nn.functional.normalize(attention, p=1, dim=-1)
                    if end_entities is not None:
                        if self.update_hidden:
                            self.agent.aggregate(candidate_nodes, edges, alpha, edges[:, -2], old_nodes_new_idx, old_nodes_old_idx, self.apply_final_red_aggr)
        # lld_scores = self.agent.compute_score(batch_size, candidate_nodes, normalize=True)
        query_embeddings = torch_scatter.scatter(support_embeddings[edges[:, 0]], edges[:, -1], dim=0, dim_size=candidate_nodes.size(0), reduce='mean')
        lld_scores = self.compute_score(batch_size, candidate_nodes, query_embeddings, normalize=True)
                    
        if not evaluate:
            if self.reward_policy == 'direct':
                self.reward = attention[batch_index, end_entities].sum().item() / batch_size
                if self.fuse_dist == 'add':
                    post_dist = (attention + lld_scores) / 2
                elif self.fuse_dist == 'mult':
                    post_dist = attention * lld_scores
                else:
                    raise NotImplementedError
                loss = self.loss_fn(post_dist, batch_size, batch_index, end_entities)
                
                # prior_loss = self.loss_fn(attention, batch_size, batch_index, end_entities)
                # lld_loss = self.loss_fn(lld_scores, batch_size, batch_index, end_entities)
                # loss = prior_loss + lld_loss
                
                return loss, entropy_loss, agent_entropy_loss
            elif self.reward_policy == 'stochastic':
                rewards = (current_entities == end_entities.unsqueeze(-1)).any(dim=-1).float()
                rewards /= current_masks.float().sum(dim=-1) + 1e-10
                self.reward = rewards.mean().item()
                if self.baseline_lambda > 0.0:
                    self.reward_baseline = (1 - self.baseline_lambda) * self.reward_baseline + \
                                           self.baseline_lambda * rewards.mean().item()
                    rewards -= self.reward_baseline
                graph_loss = (- rewards.detach() * graph_loss).mean()
                return graph_loss, entropy_loss
            else:
                raise NotImplemented
        else:
            if self.reward_policy == 'direct':
                # Unbiased evaluation protocol
                # Zhiqing Sun, Shikhar Vashishth, Soumya Sanyal, Partha P. Talukdar, Yiming Yang:
                # A Re-evaluation of Knowledge Graph Completion Methods. CoRR abs/1911.03903 (2019)
                # https://arxiv.org/pdf/1911.03903.pdf
                # TODO this will make Wiki-One slow
                # rand_idxs = list(range(self.entity_num))
                # random.shuffle(rand_idxs)
                # entity_list = torch.arange(self.entity_num, device=device)[rand_idxs]
                # attention = attention[:, rand_idxs]
                # scores, results = attention.topk(dim=-1, k=20)
                # results = entity_list[results]
                # results = results.tolist()
                # scores = scores.tolist()
                # return results, scores
                # return attention
                if self.eval_mode == 'mult':
                    return attention * lld_scores
                elif self.eval_mode == 'add':
                    return (attention + lld_scores) / 2
                if self.eval_mode == 'lld':
                    return lld_scores
                return attention
            elif self.reward_policy == 'stochastic':
                results = current_entities.tolist()
                scores = graph_loss.unsqueeze(-1).expand_as(current_entities).tolist()
                return results, scores
            else:
                raise NotImplemented
