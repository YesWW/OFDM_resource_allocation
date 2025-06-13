import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from ofdm_simulator.ofdm_simulator import OFDMSimulator
from model import Policy
from utility import Buffer, get_buffer_dataloader
import wandb
import time

class SGBS:
    def __init__(self, params_file, device):
        '''
        Initialization of our architecture and network topology.
        '''
        self._device = device
        # load configuration file
        self._config = {}
        conf_dir = Path(__file__).parents[0]
        with open(conf_dir / params_file, 'r') as f:
            self._config = yaml.safe_load(f)

        # simul params
        simul_params = {k[4:]: self._config[k] for k in self._config.keys() if k.startswith('sim.')}
        self._sim = OFDMSimulator(**simul_params)
        self._num_bs = self._sim.num_bs
        self._num_rb = self._sim.num_rb
        self._num_beam = self._sim.num_beam
        self._num_power_level = self._sim.num_power_level

        # model params
        model_params = {k[6:]: self._config[k] for k in self._config.keys() if k.startswith('model.')}
        
        # train parameters
        self._buffer_batch_size = self._config['train.buffer_batch_size']
        self._rollout_batch_size = self._config['train.rollout_batch_size']
        self._min_attn_db = self._config['train.min_attn_db']
        self._max_attn_db = self._config['train.max_attn_db']
        self._power_attn_n_cls = self._config['train.power_attn.num_level']
        self._power_attn_boundaries = torch.linspace(self._min_attn_db, self._max_attn_db, self._power_attn_n_cls).to(self._device)
        
        # model architecture
        self._ac = Policy(num_power_level=self._num_power_level, num_rb=self._num_rb, num_beam=self._num_beam,
                    power_attn_num_level=self._power_attn_n_cls, model_params=model_params, device=device)

        self._eval_file_list = self._sim.get_evaluation_networks_file_list()
        self._num_evaluation_networks = len(self._eval_file_list)
        self._eval_networks = self._sim.generate_evaluation_networks(self._num_evaluation_networks)
        self._eval_data = self._sim.generate_pyg(self._eval_networks, min_attn_db=self._min_attn_db,
                                                max_attn_db=self._max_attn_db, device=self._device)

        self._beam_width = self._config['train.beam_width']
        self._expanstion_factor = self._config['train.expansion_factor']


    def quantize_power_attn(self, g):
        '''Quantize the continous data graphs into a discrete data 
        Args:
            g_batch (graph): graph data 

        Returns:
            (graph): graph data with quantized features
        '''
        g_list = g.to_data_list()
        g2_list = []
        for g in g_list:
            # convert node power attenuation to one hot form
            node_power_attn = g.get_tensor('x').to(self._device)
            node_power_attn = torch.bucketize(node_power_attn, self._power_attn_boundaries, right=True) - 1
            node_power_attn[node_power_attn == -1] = 0
            node_power_attn = F.one_hot(node_power_attn, num_classes=self._power_attn_n_cls).to(torch.float32)
            
            # convert edge power attenuation to one hot form
            edge_power_attn = g.get_tensor('edge_attr').to(self._device)
            edge_power_attn = torch.bucketize(edge_power_attn, self._power_attn_boundaries, right=True) - 1
            valid_edge_idx = torch.all(edge_power_attn >= 0, dim=1)
            edge_power_attn = edge_power_attn[valid_edge_idx]
            edge_power_attn = F.one_hot(edge_power_attn, num_classes=self._power_attn_n_cls).to(torch.float32)
            edge_index = g.edge_index.to(self._device)
            edge_index = edge_index[:, valid_edge_idx]
            # make a new graph
            g2 = Data(x=node_power_attn, edge_index=edge_index, edge_attr=edge_power_attn)
            g2_list.append(g2)
        g2_batch = Batch.from_data_list(g2_list)
        return g2_batch
    

    def greedy_roll_out(self, g, networks, power_alloc, beam_alloc, step, link_rb_seq):
        T, B, _ = link_rb_seq.shape
        device = self._device
        g2 = self.quantize_power_attn(g)
        ptr, batch = g.ptr, g.batch

        power_alloc = power_alloc.clone()
        beam_alloc = beam_alloc.clone()

        for t in range(step, T):
            link_rb = link_rb_seq[t]  # shape [B, 2]
            link = link_rb[:, 0]      # shape [B]
            rb   = link_rb[:, 1]      # shape [B]
            valid_mask = (link != -1)

            if not valid_mask.any():
                continue  # 모든 샘플에서 step이 끝난 경우

            # 모델 추론 (전체 batch에 대해 수행)
            act_dist, _ = self._ac(
                power_alloc=power_alloc.view(-1, self._num_rb, self._num_power_level),
                beam_alloc=beam_alloc.view(-1, self._num_rb, self._num_beam),
                node_power_attn=g2.x, edge_power_attn=g2.edge_attr,
                edge_index=g2.edge_index, ptr=ptr, batch=batch
            )
            logit = act_dist.logit  # [B * N, P, B] if batched by link
            logit = logit.view(B, self._num_power_level, self._num_beam)  # [B, P, B]
            flat_logits = logit.view(B, -1)  # [B, P*B]
            greedy_idx = torch.argmax(flat_logits, dim=-1)  # [B]
            power = greedy_idx // self._num_beam
            beam = greedy_idx % self._num_beam

            # valid한 샘플만 업데이트
            for b in torch.where(valid_mask)[0]:
                l = link[b].item()
                r = rb[b].item()
                p = power[b].item()
                bm = beam[b].item()
                power_alloc[b, l, r, p] = 1.0
                beam_alloc[b, l, r, bm] = 1.0

        # rollout 완료 후 reward 계산
        rewards = []
        for b in range(B):
            l_start = ptr[b].item()
            l_end = ptr[b+1].item()
            power_sol = torch.argmax(power_alloc[b, l_start:l_end], dim=-1).cpu().numpy()
            beam_sol  = torch.argmax(beam_alloc[b, l_start:l_end], dim=-1).cpu().numpy()
            solution = {'power_level': power_sol, 'beam_index': beam_sol}
            r = self._sim.get_optimization_target(networks[b], solution)
            rewards.append(r)
        return rewards
    

    def get_lr_order(self, g):
        batch_size = len(g.ptr) - 1
        link_counts = (g.ptr[1:] - g.ptr[:-1])
        max_links = max(link_counts)
        trajectory_len = max_links * self._num_rb
        link_step_idx = torch.arange(trajectory_len, device=device)  # [T]

        valid_mask = []
        for n in link_counts.tolist():
            valid = torch.arange(trajectory_len, device=device) < (n * self._num_rb)
            valid_mask.append(valid)
        valid_mask = torch.stack(valid_mask, dim=1)  # [T, B]

        link_idx_in_batch = (link_step_idx // self._num_rb).unsqueeze(1).expand(-1, batch_size)
        rb_idx = (link_step_idx % self._num_rb).unsqueeze(1).expand(-1, batch_size)

        g_ptr_b = g.ptr[:-1].unsqueeze(0).expand(trajectory_len, -1)  # [T, B]
        global_link = g_ptr_b + link_idx_in_batch  # [T, B]

        # 패딩 위치에 -1 삽입
        global_link = torch.where(valid_mask, global_link, torch.full_like(global_link, -1))
        rb_idx = torch.where(valid_mask, rb_idx, torch.full_like(rb_idx, -1))

        link_rb_tensor = torch.stack([global_link, rb_idx], dim=2)
        return link_rb_tensor


    def train(self):
        networks = self._sim.get_networks(num_networks=self._rollout_batch_size)
        g = self._sim.generate_pyg(networks=networks, min_attn_db=self._min_attn_db, max_attn_db=self._max_attn_db, device=self._device)
        g2 = self.quantize_power_attn(g)
        link_rb = self.get_lr_order(g2)
        pass




if __name__ == '__main__':
    device = 'cuda:0'
    sgbs = SGBS(params_file='config.yaml', device=device)
    sgbs.train()
    print(1)