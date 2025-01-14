import os, itertools, functools, time
import torch
import torch.nn as nn
import networkx
import logging
from networkx.algorithms.shortest_paths.generic import shortest_path_length
from networkx import NetworkXNoPath, NetworkXError
from collections import defaultdict
import tqdm
from tensorboardX import SummaryWriter
import wandb
from contextlib import ExitStack
from pytorch_transformers import WarmupLinearSchedule

from src.grapher import KG
from src.torch_utils import load_embedding
from src.utils import unserialize, serialize, inverse_relation, load_facts, load_index, translate_facts, \
    add_reverse_relations, TqdmLoggingHandler
from src.trainer import Trainer
from src.module import CogKR, Summary
from src.evaluation import hitRatio, MAP, multi_mean_measure, MRR


class Main:
    def __init__(self, args, root_directory, device=torch.device("cpu"), comment="", sparse_embed=False):
        self.args = args
        self.use_wandb = args.use_wandb
        self.root_directory = root_directory
        self.comment = comment
        self.device = device
        self.config = {
            "trainer": {"evaluate_inverse": False},
            'train': {
                'entropy_beta': 0.0,
                'validate_metric': 'MRR'
            },
            "pretrain": {
                "keep_embed": False,
                "log_interval": 1000,
                "evaluate_interval": 10000
            },
            'model': {
                'message': True,
                'entity_embed': True
            },
            "sparse_embed": False,
            "optimizer": {
                "scheduler": {
                    "factor": 0.6,
                    "patience": 5,
                    "min_lr": 1e-6
                }
            }
        }
        self.measure_dict = {
            'Hit1': functools.partial(hitRatio, topn=1),
            'Hit3': functools.partial(hitRatio, topn=3),
            'Hit5': functools.partial(hitRatio, topn=5),
            'Hit10': functools.partial(hitRatio, topn=10),
            # 'hitRatio': hitRatio,
            'MRR': MRR
        }
        self.sparse_embed = sparse_embed
        self.best_results = {}
        self.steps_no_improve = 0
        self.data_loaded = False
        self.env_built = False
        self.sample_action = args.sample_action
        self.eval_mode = args.eval_mode

    def update_config(self, config: dict):
        for key, value in config.items():
            if isinstance(value, dict):
                if key in self.config:
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            else:
                self.config[key] = value

    def init(self, config=None):
        if config is not None:
            self.config = config
        if not self.data_loaded:
            self.load_data()
        if not self.env_built:
            self.build_env(self.config['graph'])
        self.build_model(self.config['model'])
        self.build_logger()
        self.build_optimizer(self.config['optimizer'])

    def load_data(self):
        self.data_directory = os.path.join(self.root_directory, "data")
        self.entity_dict = load_index(os.path.join(self.data_directory, "ent2id.txt"))
        self.relation_dict = load_index(os.path.join(self.data_directory, "relation2id.txt"))
        self.facts_data = translate_facts(load_facts(os.path.join(self.data_directory, "train.txt")), self.entity_dict,
                                          self.relation_dict)
        test_support = os.path.join(self.data_directory, "test_support.txt")
        if os.path.exists(test_support):
            self.test_support = translate_facts(load_facts(test_support), self.entity_dict, self.relation_dict)
        else:
            self.test_support = []
        valid_support = os.path.join(self.data_directory, "valid_support.txt")
        if os.path.exists(valid_support):
            self.valid_support = translate_facts(load_facts(valid_support), self.entity_dict, self.relation_dict)
        else:
            self.valid_support = []
        test_eval = os.path.join(self.data_directory, "test_eval.txt")
        if not os.path.exists(test_eval):
            test_eval = os.path.join(self.data_directory, "test.txt")
        # test_eval = load_facts(test_eval)
        tail_test_eval = load_facts(test_eval)
        head_test_eval = add_reverse_relations(tail_test_eval)
        # if self.config["trainer"]["evaluate_inverse"]:
        #     test_eval = add_reverse_relations(test_eval)
        #     print(f"Test facts {len(test_eval)} after adding inverse")
        # self.test_eval = translate_facts(test_eval, self.entity_dict, self.relation_dict)
        self.tail_test_eval = translate_facts(tail_test_eval, self.entity_dict, self.relation_dict)
        self.head_test_eval = translate_facts(head_test_eval, self.entity_dict, self.relation_dict)
        
        valid_eval = os.path.join(self.data_directory, "valid_eval.txt")
        if not os.path.exists(valid_eval):
            valid_eval = os.path.join(self.data_directory, "valid.txt")
        tail_valid_eval = load_facts(valid_eval)
        head_valid_eval = add_reverse_relations(tail_valid_eval)
        # if self.config["trainer"]["evaluate_inverse"]:
        #     valid_eval = add_reverse_relations(valid_eval)
        # self.valid_eval = translate_facts(valid_eval, self.entity_dict, self.relation_dict)
        self.tail_valid_eval = translate_facts(tail_valid_eval, self.entity_dict, self.relation_dict)
        self.head_valid_eval = translate_facts(head_valid_eval, self.entity_dict, self.relation_dict)
        
        # augment
        with open(os.path.join(self.data_directory, 'pagerank.txt')) as file:
            self.pagerank = list(map(lambda x: float(x.strip()), file.readlines()))
        if os.path.exists(os.path.join(self.data_directory, "fact_dist")):
            self.fact_dist = unserialize(os.path.join(self.data_directory, "fact_dist"))
        else:
            self.fact_dist = None
        if os.path.exists(os.path.join(self.data_directory, "rel2candidates")):
            self.rel2candidate = unserialize(os.path.join(self.data_directory, "rel2candidates"))
        else:
            self.rel2candidate = {}
        self.id2entity = sorted(self.entity_dict.keys(), key=self.entity_dict.get)
        self.id2relation = sorted(self.relation_dict.keys(), key=self.relation_dict.get)
        self.data_loaded = True

    def build_env(self, graph_config, build_matrix=True):
        self.config['graph'] = graph_config
        self_loop_id = None
        if graph_config['add_self_loop']:
            self.relation_dict['SELF_LOOP'] = len(self.relation_dict)
            self_loop_id = self.relation_dict['SELF_LOOP']
        self.reverse_relation = [self.relation_dict[inverse_relation(relation)] for relation in self.id2relation]
        self.kg = KG(self.facts_data, entity_num=len(self.entity_dict), relation_num=len(self.relation_dict),
                     node_scores=self.pagerank, build_matrix=build_matrix, self_loop_id=self_loop_id, **graph_config)
        self.trainer = Trainer(self.kg, self.facts_data, reverse_relation=self.reverse_relation, cutoff=3,
                               validate_tasks=(self.valid_support, self.tail_valid_eval, self.head_valid_eval),
                               test_tasks=(self.test_support, self.tail_test_eval, self.head_test_eval),
                               id2entity=self.id2entity, id2relation=self.id2relation, rel2candidate=self.rel2candidate,
                               fact_dist=self.fact_dist, **self.config['trainer'])
        self.env_built = True

    def load_model(self, path, batch_id):
        """
        remain for compatible
        """
        config_path = os.path.join(path, 'config,json')
        if os.path.exists(config_path):
            self.config = unserialize(os.path.join(path, 'config.json'))
            self.cogKR = CogKR(graph=self.kg, entity_dict=self.entity_dict, relation_dict=self.relation_dict,
                               device=self.device, **self.config['model']).to(self.device)
            model_state = torch.load(os.path.join(path, str(batch_id) + ".model.dict"))
            self.cogKR.load_state_dict(model_state)

    def load_state(self, path, train=True):
        state = torch.load(path, map_location=self.device)
        if 'config' in state:
            self.config = state['config']
        self.build_model(self.config['model'])
        self.cogKR.load_state_dict(state['model'])
        if train:
            self.build_optimizer(self.config['optimizer'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.log_directory = os.path.dirname(path)
            if 'batch_id' in state:
                self.batch_id = state['batch_id']
            else:
                self.batch_id = int(os.path.basename(path).split('.')[0])
            self.build_logger(self.log_directory, self.batch_id)
            if 'best_results' in state:
                self.best_results = state['best_results']
            else:
                self.best_results = {}
            self.total_graph_loss = state.get('graph_loss', 0.0)
            self.total_graph_size = state.get('graph_size', 0.0)
            self.total_reward = state.get('reward', 0.0)
            self.steps_no_improve = state.get('steps_no_improve', 0)

    def load_pretrain(self, path):
        state_dict = torch.load(path)
        self.cogKR.summary.load_state_dict(state_dict)

    def build_pretrain_model(self, model_config):
        self.config['model'] = model_config
        entity_embeddings = nn.Embedding(len(self.entity_dict) + 1, model_config['embed_size'],
                                         padding_idx=len(self.entity_dict))
        relation_embeddings = nn.Embedding(len(self.relation_dict) + 1, model_config['embed_size'],
                                           padding_idx=len(self.relation_dict))
        self.summary = Summary(model_config['hidden_size'], graph=self.kg,
                               entity_embeddings=entity_embeddings, relation_embeddings=relation_embeddings).to(
            self.device)

    def build_model(self, model_config):
        self.config['model'] = model_config
        self.stochastic_policy = model_config['reward_policy'] == 'stochastic'
        self.cogKR = CogKR(graph=self.kg, entity_dict=self.entity_dict, relation_dict=self.relation_dict,
                           device=self.device, sparse_embed=self.sparse_embed, id2entity=self.id2entity,
                           id2relation=self.id2relation,
                           use_summary=self.config['trainer']['meta_learn'], **model_config).to(self.device)
        self.agent = self.cogKR.agent
        self.coggraph = self.cogKR.cog_graph
        # self.summary = self.cogKR.summary

    def build_pretrain_optimiaer(self, optimizer_config):
        self.config['pretrain_optimizer'] = optimizer_config
        if self.config['pretrain']['keep_embed']:
            print("Keep embedding")
            self.summary.entity_embeddings.weight.requires_grad_(False)
            self.summary.relation_embeddings.weight.requires_grad_(False)
        self.parameters = list(filter(lambda x: x.requires_grad, self.summary.parameters()))
        self.optimizer = torch.optim.__getattribute__(optimizer_config['name'])(self.parameters,
                                                                                **optimizer_config['config'])
        self.loss_function = nn.CrossEntropyLoss()
        self.predict_loss = 0.0

    def build_optimizer(self, optimizer_config):
        self.config['optimizer'] = optimizer_config
        parameter_ids = set()
        self.parameters = []
        self.optim_params = []
        sparse_ids = set()
        # self.embed_parameters = list(self.cogKR.relation_embeddings.parameters()) + list(
        #     self.cogKR.entity_embeddings.parameters())
        # self.parameters.extend(self.embed_parameters)
        # parameter_ids.update(map(id, self.embed_parameters))
        # print('Embedding parameters:', list(map(lambda x: x.size(), self.embed_parameters)))
        # if self.sparse_embed:
        #     self.embed_optimizer = torch.optim.SparseAdam(self.embed_parameters, **optimizer_config['embed'])
        # else:
        #     self.optim_params.append({'params': self.embed_parameters, **optimizer_config['embed']})
        # self.summary_parameters = list(filter(lambda x: id(x) not in parameter_ids, self.summary.parameters()))
        # self.parameters.extend(self.summary_parameters)
        # parameter_ids.update(map(id, self.summary_parameters))
        self.agent_parameters = list(filter(lambda x: id(x) not in parameter_ids, self.cogKR.parameters()))
        self.parameters.extend(self.agent_parameters)
        # if self.sparse_embed:
        #     sparse_ids.update(map(id, self.cogKR.entity_embeddings.parameters()))
        #     sparse_ids.update(map(id, self.cogKR.relation_embeddings.parameters()))
        self.dense_parameters = list(filter(lambda x: id(x) not in sparse_ids, self.parameters))
        print(list(map(lambda x: x.size(), self.dense_parameters)))
        self.optim_params.extend([{'params': self.agent_parameters, **optimizer_config['agent']}])
        self.optimizer = torch.optim.__getattribute__(optimizer_config['name'])(self.optim_params,
                                                                                **optimizer_config['config'])
        if optimizer_config["scheduler_config"]["name"] == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', verbose=True,
                                                                        **optimizer_config['scheduler'])
            print('Using ReduceLROnPlateau schduler ...')
        else:
            self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=optimizer_config["scheduler_config"]["warmup_steps"], 
                                                  t_total=self.config['train']['max_steps'])
            print('Using WarmupLinearSchedule schduler ...')
        self.total_graph_loss = 0.0
        self.total_graph_size, self.total_reward = 0, 0.0
        self.entropy_beta = self.config['train']['entropy_beta']
        self.agent_entropy_beta = self.config['train']['agent_entropy_beta']
        if self.stochastic_policy:
            print("Entropy beta: ", self.entropy_beta)

    def save_state(self, is_best=False):
        if is_best:
            filename = os.path.join(self.log_directory, "best.state")
        else:
            filename = os.path.join(self.log_directory, "latest.state")
        torch.save({
            'config': self.config,
            'model': self.cogKR.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'batch_id': self.batch_id + 1,
            'graph_loss': self.total_graph_loss,
            'reward': self.total_reward,
            'graph_size': self.total_graph_size,
            'log_file': self.summary_dir,
            'best_results': self.best_results,
            'steps_no_improve': self.steps_no_improve
        }, filename)

    def evaluate_model(self, mode='test', part='tail', output=None, save_correct_cases=None, save_graph=None, **kwargs):
        with torch.no_grad():
            self.cogKR.eval()
            current = time.time()
            results = multi_mean_measure(
                self.trainer.evaluate_generator(self.cogKR, self.trainer.evaluate(mode=mode, part=part, **kwargs),
                                                save_result=output,
                                                save_graph=save_graph, eval_mode=self.eval_mode,
                                                batch_size=self.config['train']['test_batch_size']), 
                self.measure_dict)

            if save_correct_cases:
                with open(save_correct_cases, 'w', encoding='utf-8') as fout:
                    for triple in self.trainer.correct_cases:
                        fout.write(triple + '\n')
            
            if self.args.inference_time:
                print(time.time() - current)
            self.cogKR.train()
        return results

    def build_logger(self, log_directory=None, batch_id=None):
        self.log_directory = log_directory
        if self.log_directory is None:
            self.log_directory = os.path.join(self.root_directory, "log",
                                              "{}-{}-{}-{}-{}-{}-{}-{}".format(time.strftime("%m-%d-%H"),
                                                                         self.config['model']['max_steps'],
                                                                         '_'.join([str(_) for _ in self.config['model']['topk']]),
                                                                         self.config['graph']['train_width'],
                                                                         self.config['model']['agent_num'],
                                                                         self.config['model']['reward_policy'],
                                                                         self.config['model']['message'], self.comment))
            if not os.path.exists(self.log_directory):
                os.makedirs(self.log_directory)
            serialize(self.config, os.path.join(self.log_directory, 'config.json'), in_json=True)
        self.summary_dir = os.path.join(self.log_directory, "log")
        if batch_id is None:
            self.writer = SummaryWriter(self.summary_dir)
            self.batch_sampler = itertools.count()
        else:
            self.writer = SummaryWriter(self.summary_dir, purge_step=batch_id)
            self.batch_sampler = itertools.count(start=batch_id)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(TqdmLoggingHandler())
        logfile = os.path.join(self.log_directory, "log.txt")
        if batch_id is None:
            logfile = logging.FileHandler(logfile, "w")
        else:
            logfile = logging.FileHandler(logfile, "a")
        self.logger.addHandler(logfile)
        if self.use_wandb:
            wandb.init(
                project="graph_reasoning",
                name="{}-{}-{}-{}-{}-{}".format(time.strftime("%m-%d-%H"), 
                                                self.config['model']['max_steps'], 
                                                self.config['model']['topk'], 
                                                self.config['model']['reward_policy'], 
                                                self.config['model']['message'], self.comment),
                config=self.config
            )

    def pretrain(self, single_step=False):
        for batch_id in tqdm(self.batch_sampler):
            support_pairs, labels = self.trainer.pretrain_sample(self.config['pretrain']['batch_size'])
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)
            scores = self.summary(support_pairs, predict=True)
            loss = self.loss_function(scores, labels)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, 0.25, norm_type='inf')
            self.optimizer.step()

            self.predict_loss += loss.item()
            if (batch_id + 1) % self.config['pretrain']['log_interval'] == 0:
                self.logger.info("predict loss: {}".format(self.predict_loss))
                self.writer.add_scalar('predict_loss', self.predict_loss / 1000, batch_id)
                self.predict_loss = 0.0
            if (batch_id + 1) % self.config['pretrain']['evaluate_interval'] == 0:
                torch.save(self.summary.state_dict(), os.path.join(self.log_directory,
                                                                   'summary.' + str(batch_id + 1) + ".dict"))
            if single_step:
                break

    def log(self):
        interval = self.config['train']['log_interval']
        self.writer.add_scalar('graph_loss', self.total_graph_loss / interval, self.batch_id)
        print("graph loss: {}".format(self.total_graph_loss / interval))
        self.writer.add_scalar('reward', self.total_reward / interval, self.batch_id)
        print("reward: {}".format(self.total_reward / interval))
        self.writer.add_scalar('graph_size', self.total_graph_size / interval, self.batch_id)
        print("graph size: {}".format(self.total_graph_size / interval))
        self.logger.info("[Step: {}] Loss: {}, Reward: {}".format(self.batch_id + 1, self.total_graph_loss, self.total_reward / interval))
        print("[Step: {}] Loss: {}, Reward: {}".format(self.batch_id + 1, self.total_graph_loss, self.total_reward / interval))
        self.total_graph_loss, self.total_graph_size, self.total_reward = 0.0, 0, 0.0

    def train(self, single_step=False):
        MB = 1024.0* 1024.0
        meta_learn = self.config['trainer']['meta_learn']
        validate_metric = self.config['train']['validate_metric']
        for self.batch_id in tqdm.tqdm(self.batch_sampler):
            if 'max_steps' in self.config['train'] and self.batch_id > self.config['train']['max_steps']:
                break
            support_pairs, query_heads, query_tails, relations, other_correct_answers = self.trainer.sample(
                self.config['train']['batch_size'])
            if meta_learn:
                graph_loss, entropy_loss, agent_entropy_loss = self.cogKR(query_heads, other_correct_answers, end_entities=query_tails, 
                                                                          support_pairs=support_pairs, evaluate=False)
            else:
                graph_loss, entropy_loss, agent_entropy_loss = self.cogKR(query_heads, other_correct_answers, end_entities=query_tails, 
                                                                          relations=relations, evaluate=False, do_sample=self.sample_action, eval_mode=self.eval_mode)
            self.optimizer.zero_grad()
            if self.sparse_embed:
                self.embed_optimizer.zero_grad()
            (graph_loss - self.entropy_beta * entropy_loss + self.agent_entropy_beta * agent_entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.dense_parameters, 0.25, norm_type='inf')
            self.optimizer.step()
            if self.sparse_embed:
                self.embed_optimizer.step()
            if torch.isnan(graph_loss):
                print('nan errors...')
                break
            else:
                self.total_graph_loss += graph_loss.item()
                self.total_reward += self.cogKR.reward
            if self.use_wandb:
                wandb.log({
                    "graph_loss": graph_loss.item(),
                    "reward": self.cogKR.reward,
                    "entropy_loss": entropy_loss if isinstance(entropy_loss, float) else entropy_loss.item(),
                })
            if self.stochastic_policy and (self.batch_id + 1) % 200 == 0:
                self.entropy_beta *= 0.9
                self.logger.info("Beta decay to: {}".format(self.entropy_beta))
            if (self.batch_id + 1) % self.config['train']['log_interval'] == 0:
                self.log()
                self.logger.info('using VM: {}'.format(torch.cuda.max_memory_allocated()/ MB))
            if (self.batch_id + 1) >= self.config['train']['start_eval_step'] and (self.batch_id + 1) % self.config['train']['evaluate_interval'] == 0:
                self.save_state(is_best=False)
                with torch.no_grad():
                    # for fuse_dist in ['prior', 'lld']:
                    tail_validate_results = self.evaluate_model(mode='valid', part='tail')
                    self.logger.info("Validate tail results: {}".format(tail_validate_results))
                    # head_validate_results = self.evaluate_model(mode='valid', part='head')
                    # self.logger.info("Validate head results: {}".format(head_validate_results))
                    # validate_results = {key: (tail_validate_results[key] + head_validate_results[key]) / 2 for key in tail_validate_results}
                    validate_results = tail_validate_results
                    self.logger.info("Validate results: {}".format(validate_results))
                    if self.use_wandb:
                        wandb.log(validate_results)
                    old_lr = self.optimizer.param_groups[0]["lr"]
                    self.scheduler.step(validate_results[validate_metric])
                    new_lr = self.optimizer.param_groups[0]["lr"]
                    if old_lr != new_lr:
                        self.writer.add_scalar('current_lr', new_lr, self.batch_id)
                        print("change learning rate from {} to {}...".format(old_lr, new_lr))
                    if self.use_wandb:
                        wandb.log({
                            "lr": new_lr
                        })
                    if validate_metric not in self.best_results or validate_results[validate_metric] >= \
                            self.best_results[
                                validate_metric]:
                        # tail_test_results = self.evaluate_model(mode='test', part='tail')
                        # self.logger.info("Test tail results: {}".format(tail_test_results))
                        # head_test_results = self.evaluate_model(mode='test', part='head')
                        # self.logger.info("Test head results: {}".format(head_test_results))
                        # test_results = {key: (tail_test_results[key] + head_test_results[key]) / 2 for key in tail_test_results}
                        # for key, value in test_results.items():
                        #     self.writer.add_scalar(key, value, self.batch_id)
                        # self.logger.info("Test results: {}".format(test_results))
                        self.save_state(is_best=True)
                        self.steps_no_improve = 0
                    else:
                        self.steps_no_improve += self.config['train']['evaluate_interval']
                    for key, value in validate_results.items():
                        if key not in self.best_results or value > self.best_results[key]:
                            self.best_results[key] = value
                if self.steps_no_improve > self.config['train']['max_wait_step']:
                    self.logger.info(f"Training stops after no improvement step {self.steps_no_improve}")
                    return
            self.local = locals()
            if single_step:
                break

    def get_fact_dist(self, ignore_relation=True):
        graph = self.kg.to_networkx(multi=True, neighbor_limit=256)
        fact_dist = {}
        for relation, pairs in tqdm.tqdm(self.trainer.train_query.items()):
            deleted_edges = []
            if ignore_relation:
                reverse_relation = self.reverse_relation[relation]
                for head, tail in itertools.chain(pairs, self.trainer.train_support[relation],
                                                  self.trainer.train_query[reverse_relation],
                                                  self.trainer.train_support[reverse_relation]):
                    try:
                        graph.remove_edge(head, tail, relation)
                        deleted_edges.append((head, tail, relation))
                    except NetworkXError:
                        pass
                    try:
                        graph.remove_edge(head, tail, reverse_relation)
                        deleted_edges.append((head, tail, reverse_relation))
                    except NetworkXError:
                        pass
            for head, tail in itertools.chain(self.trainer.train_query[relation], self.trainer.train_support[relation]):
                delete_edge = False
                try:
                    graph.remove_edge(head, tail, relation)
                    delete_edge = True
                except NetworkXError:
                    pass
                try:
                    dist = shortest_path_length(graph, head, tail)
                except NetworkXNoPath or KeyError:
                    dist = -1
                fact_dist[(head, relation, tail)] = dist
                if delete_edge:
                    graph.add_edge(head, tail, relation)
            graph.add_edges_from(deleted_edges)
        return fact_dist

    def get_dist_dict(self, mode='test'):
        self.graph = self.kg.to_networkx(multi=False)
        global_dist_count = defaultdict(int)
        fact_dist = {}
        data_loader = self.trainer.evaluate(mode=mode)
        for relation, support_pair, evaluate_facts, *other in data_loader:
            dist_count = defaultdict(int)
            for head, tail in evaluate_facts:
                try:
                    dist = shortest_path_length(self.graph, head, tail)
                except networkx.NetworkXNoPath:
                    dist = -1
                dist_count[dist] += 1
                global_dist_count[dist] += 1
                fact_dist[(self.id2entity[head], self.id2relation[relation], self.id2entity[tail])] = dist
        print(sorted(global_dist_count.items(), key=lambda x: x[0]))
        return fact_dist

    def get_onehop_ratio(self):
        e1e2_rel = {}
        for relation, pairs in self.trainer.train_support.items():
            for pair in pairs:
                e1e2_rel.setdefault(pair, set())
                e1e2_rel[pair].add(relation)

        sums, num = 0, 0
        for relation, pairs in self.trainer.test_ground.items():
            print(relation)
            for head, tail in pairs:
                num += 1
                if (head, tail) in e1e2_rel:
                    sums += 1
                    print(e1e2_rel[(head, tail)])
        return sums / num

    def get_test_fact_num(self):
        sums = 0
        for task in self.trainer.test_relations:
            sums += len(self.trainer.test_ground[task])
        return sums

    def save_to_hyper(self, data_dir):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        def save_to_file(data, path):
            with open(path, "w") as output:
                for head, relation, tail in data:
                    output.write("{}\t{}\t{}\n".format(head, relation, tail))

        def save_dict(data, path):
            with open(path, "w") as output:
                for entry, idx in data.items():
                    output.write("{}\t{}\n".format(idx, entry))

        facts_data = list(filter(lambda x: not self.id2relation[x[1]].endswith("_inv"), self.facts_data))
        facts_data = list(
            map(lambda x: (self.id2entity[x[0]], self.id2relation[x[1]], self.id2entity[x[2]]), facts_data))
        supports = [(self.id2entity[head], self.id2relation[relation], self.id2entity[tail]) for relation, (head, tail)
                    in
                    self.trainer.test_support.items()]
        facts_data = list(itertools.chain(facts_data, supports))
        valid_evaluate = [(self.id2entity[head], self.id2relation[relation], self.id2entity[tail]) for relation in
                          self.trainer.validate_relations for head, tail in self.trainer.test_ground[relation]]
        test_evaluate = [(self.id2entity[head], self.id2relation[relation], self.id2entity[tail]) for relation in
                         self.trainer.test_relations for head, tail in
                         self.trainer.test_ground[relation]]
        save_to_file(itertools.chain(facts_data, *itertools.repeat(supports, 1)), os.path.join(data_dir, 'train.txt'))
        save_to_file(valid_evaluate, os.path.join(data_dir, "dev.txt"))
        save_to_file(test_evaluate, os.path.join(data_dir, "test.txt"))
        save_to_file(add_reverse_relations(facts_data), os.path.join(data_dir, "graph.txt"))
        save_dict(self.entity_dict, os.path.join(data_dir, 'entities.dict'))
        save_dict(self.relation_dict, os.path.join(data_dir, 'relations.dict'))

    def save_to_multihop(self, data_dir):
        """
        generate the file in the format of https://github.com/salesforce/MultiHopKG
        """
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        def save_to_file(data, path):
            with open(path, "w") as output:
                for head, relation, tail in data:
                    output.write("{}\t{}\t{}\n".format(head, tail, relation))

        facts_data = list(filter(lambda x: not self.id2relation[x[1]].endswith("_inv"), self.facts_data))
        facts_data = list(
            map(lambda x: (self.id2entity[x[0]], self.id2relation[x[1]], self.id2entity[x[2]]), facts_data))
        supports = [(self.id2entity[head], relation, self.id2entity[tail]) for relation, (head, tail) in
                    self.trainer.test_support.items()]
        valid_evaluate = [(self.id2entity[head], relation, self.id2entity[tail]) for relation in
                          self.trainer.validate_relations for head, tail in self.trainer.test_ground[relation]]
        test_evaluate = [(self.id2entity[head], relation, self.id2entity[tail]) for relation in
                         self.trainer.test_relations for head, tail in
                         self.trainer.test_ground[relation]]
        save_to_file(itertools.chain(facts_data, supports), os.path.join(data_dir, 'raw.kb'))
        save_to_file(itertools.chain(facts_data, supports), os.path.join(data_dir, "train.triples"))
        save_to_file(valid_evaluate, os.path.join(data_dir, "dev.triples"))
        save_to_file(test_evaluate, os.path.join(data_dir, "test.triples"))
        with open(os.path.join(data_dir, 'raw.pgrk'), "w") as output:
            for key, value in enumerate(self.pagerank):
                output.write("{}\t:{}\n".format(self.id2entity[key], value))
        with open(os.path.join(data_dir, "rel2candidates"), "w") as output:
            for relation, candidates in self.rel2candidate.items():
                for candidate in candidates:
                    output.write("{}\t{}\n".format(relation, self.id2entity[candidate]))


if __name__ == "__main__":
    # Parse arguments
    from src.parse_args import args

    if args.gpu is not None and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device("cpu")
    torch.set_num_threads(args.num_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_threads)
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
    main_body = Main(args, root_directory=args.directory, device=device, comment=args.comment)
    if args.config:
        main_body.update_config(unserialize(args.config))
    print(main_body.config)
    main_body.sparse_embed = main_body.config['sparse_embed']
    main_body.load_data()
    main_body.build_env(main_body.config['graph'])
    if args.save_minerva:
        data_dir = os.path.join(args.directory, "minerva")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        main_body.save_to_hyper(data_dir)
    elif args.get_fact_dist:
        fact_dist = main_body.get_fact_dist(main_body.config['trainer']['ignore_relation'])
        serialize(fact_dist, os.path.join(main_body.data_directory, "fact_dist"))
    elif args.get_dist_dict:
        dist_dict = main_body.get_dist_dict(mode='test')
        serialize(dist_dict, os.path.join(main_body.data_directory, "dist_dict"))
    elif args.pretrain:
        main_body.build_pretrain_model(main_body.config['model'])
        main_body.build_pretrain_optimiaer(main_body.config['optimizer'])
        if args.load_embed:
            entity_embed_path = os.path.join(args.directory, ".".join(('entity2vec', args.load_embed)))
            relation_embed_path = os.path.join(args.directory, ".".join(('relation2vec', args.load_embed)))
            print("Load Entity Embeddings from {}".format(entity_embed_path))
            print("Load Relation Embeddings from {}".format(relation_embed_path))
            load_embedding(main_body.summary, entity_embed_path, relation_embed_path)
        main_body.build_logger()
        main_body.pretrain()
    else:
        if args.inference:
            if os.path.isdir(args.load_state):
                entries = list(filter(lambda x: x.endswith('.state'), os.listdir(args.load_state)))
                # entries = sorted(entries, key=lambda x: int(x.split('.')[0]))
                main_body.build_logger(args.log_dir, batch_id=0)
                for entry in entries:
                    if entry == 'best.state':
                        # batch_id = int(entry.split('.')[0])
                        # print(batch_id)
                        state_path = os.path.join(args.load_state, entry)
                        main_body.load_state(state_path, train=False)
                        main_body.eval_mode = args.eval_mode
                        # tail_results = main_body.evaluate_model(mode='test', part='tail', save_graph=os.path.join(args.load_state, 'tail_save_graph.json'))
                        # head_results = main_body.evaluate_model(mode='test', part='head', save_graph=os.path.join(args.load_state, 'head_save_graph.json'))
                        # tail_results = main_body.evaluate_model(mode='test', part='tail', save_correct_cases=os.path.join(args.load_state, 'tail_correct_cases.txt'))
                        # head_results = main_body.evaluate_model(mode='test', part='head', save_correct_cases=os.path.join(args.load_state, 'head_correct_cases.txt'))
                        tail_results = main_body.evaluate_model(mode='test', part='tail')
                        head_results = main_body.evaluate_model(mode='test', part='head')
                        main_body.logger.info("Test tail results: {}".format(tail_results))
                        main_body.logger.info("Test head results: {}".format(head_results))
                        results = {key: (tail_results[key] + head_results[key]) / 2 for key in tail_results}
                        main_body.logger.info("Test results: {}".format(results))
                        for key, value in results.items():
                            main_body.writer.add_scalar(key, value, 0)
            else:
                if args.load_state:
                    main_body.load_state(args.load_state, train=False)
                # print("Evaluate on valid data")
                # results = main_body.evaluate_model(mode='valid')
                # print(results)
                print("Evaluate on test data")
                if args.save_result:
                    save_result = os.path.join(os.path.dirname(args.load_state), args.save_result)
                else:
                    save_result = None
                if args.save_graph:
                    save_graph = os.path.join(os.path.dirname(args.load_state), args.save_graph)
                else:
                    save_graph = None
                results = main_body.evaluate_model(mode='test', output=save_result, save_graph=save_graph)
                print(results)
        else:
            if args.load_state:
                main_body.load_state(args.load_state, train=not args.inference)
            else:
                main_body.build_model(main_body.config['model'])
                if args.load_pretrain:
                    main_body.load_pretrain(args.load_pretrain)
                main_body.build_logger()
                main_body.build_optimizer(main_body.config['optimizer'])
                if args.load_embed:
                    entity_embed_path = os.path.join(args.directory, "data",
                                                     ".".join(('entity2vec', args.load_embed)))
                    relation_embed_path = os.path.join(args.directory, "data",
                                                       ".".join(('relation2vec', args.load_embed)))
                    main_body.logger.info("Load Entity Embeddings from {}".format(entity_embed_path))
                    main_body.logger.info("Load Relation Embeddings from {}".format(relation_embed_path))
                    load_embedding(main_body.cogKR, entity_embed_path, relation_embed_path)
            with ExitStack() as stack:
                # stack.callback(main_body.save_state)
                main_body.train()
    
    if args.use_wandb:
        wandb.finish()
