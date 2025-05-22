import numpy as np
import torch
from pathlib import Path
import time
import pandas as pd
import yaml

import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from ofdm_simulator.ofdm_simulator_4 import OFDMSimulator
from model import AC

class RLSolverOFDM:
    def __init__(self, gpu=True, params_file='config.yaml', device='cpu', max_iter=1000):
        self._device = device
        self._max_iter = max_iter

        # config load
        conf_dir = Path(__file__).parents[0]
        with open(conf_dir / params_file, 'r') as f:
            self._config = yaml.safe_load(f)

        # simulator init
        simul_params = {k[4:]: self._config[k] for k in self._config if k.startswith('sim.')}
        self._sim = OFDMSimulator(**simul_params)

        self._num_rb = self._sim.num_rb
        self._num_beam = self._sim.num_beam
        self._num_power_level = self._sim.num_power_level
        self._min_attn_db = self._config['train.min_attn_db']
        self._max_attn_db = self._config['train.max_attn_db']
        self._power_attn_n_cls = self._config['train.power_attn.num_level']
        self._power_attn_boundaries = torch.linspace(self._min_attn_db, self._max_attn_db, self._power_attn_n_cls).to(self._device)

        self._eval_file_list = self._sim.get_evaluation_networks_file_list()
        self._num_evaluation_networks = len(self._eval_file_list)

        self._ac = AC(num_power_level=self._num_power_level, num_rb=self._num_rb,
                      num_beam=self._num_beam, power_attn_num_level=self._power_attn_n_cls, model_params={k[6:]: self._config[k] for k in self._config if k.startswith('model.')},
                      device=device).to(device)



    def solve_all_evaluation_networks(self):
        perf_list = []
        elapsed_time_list = []
        for idx in range(self._num_evaluation_networks):
            perf, elapsed_time, _ = self.solve_evaluation_network(idx)
            perf_list.append(perf)
            elapsed_time_list.append(elapsed_time)
        avg_perf = sum(perf_list) / len(perf_list)
        avg_elapsed_time = sum(elapsed_time_list) / len(elapsed_time_list)
        return avg_perf, avg_elapsed_time

    def solve_evaluation_network(self, index):
        network = self._sim.get_evaluation_network(self._eval_file_list[index])
        perf, elapsed_time, solution = self.solve(network)
        return perf, elapsed_time, solution


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


    def solve(self, networks):
        start = time.perf_counter()
        self._ac.eval()
        if len(networks) == 3:
            networks = [networks] * 4
        batch = self._sim.generate_pyg(networks, min_attn_db=self._min_attn_db, max_attn_db=self._max_attn_db, device=self._device)
        batch = self.quantize_power_attn(batch)
        ptr = batch.ptr
        batch_idx = batch.batch
        batch_size = len(ptr) - 1
        num_node = batch.x.shape[0]

        power_alloc = torch.zeros((num_node, self._num_rb, self._num_power_level), device=self._device)
        beam_alloc = torch.zeros((num_node, self._num_rb, self._num_beam), device=self._device)
        power_alloc[:, :, 0] = 1  # default initialization

        unterminated_node = torch.full((num_node, self._num_rb), True, device=self._device)
        ongoing = torch.full((batch_size,), True, device=self._device)

        best_target, best_solution = -np.inf, None
        output_log = []

        for it in range(self._max_iter):
            with torch.no_grad():
                actions = self._ac(power_alloc=power_alloc, beam_alloc=beam_alloc,
                                       node_power_attn=batch.x, edge_power_attn=batch.edge_attr,
                                       edge_index=batch.edge_index, ptr=ptr, batch=batch_idx)

                # actions = act_dist.sample()
                for idx, act in enumerate(actions):
                    if not ongoing[idx]:
                        continue
                    link = act // (self._num_rb * self._num_beam)
                    rbb = act % (self._num_rb * self._num_beam)
                    rb = rbb // self._num_beam
                    beam = rbb % self._num_beam
                    #link, rb, beam = act
                    ptr_link = ptr[idx] + link

                    current_power = torch.argmax(power_alloc[ptr_link, rb]).item()
                    if current_power < self._num_power_level - 1:
                        power_alloc[ptr_link, rb] = 0
                        power_alloc[ptr_link, rb, current_power + 1] = 1

                    beam_alloc[ptr_link, rb] = 0
                    beam_alloc[ptr_link, rb, beam] = 1

                    unterminated_node[ptr_link, rb] = False
                    ongoing[idx] = torch.any(unterminated_node[ptr[idx]:ptr[idx+1]])

            batch_targets = []
            for idx in range(batch_size):
                power_sol = torch.argmax(power_alloc[ptr[idx]:ptr[idx+1]], dim=-1).cpu().numpy()
                beam_sol = torch.argmax(beam_alloc[ptr[idx]:ptr[idx+1]], dim=-1).cpu().numpy()
                solution = {'power_level': power_sol, 'beam_index': beam_sol}
                target = self._sim.get_optimization_target(networks[idx], solution)
                batch_targets.append(target)

            avg_target = np.mean(batch_targets)
            if avg_target > best_target:
                best_target = avg_target
                best_solution = (power_alloc.clone(), beam_alloc.clone())

            #elapsed = time.perf_counter() - start
            #output_log.append([it, best_target, elapsed])
            #print(f"Iter: {it}, Target: {best_target:.4f}, Time: {elapsed:.2f}s")

        # df = pd.DataFrame(output_log, columns=['iter', 'target', 'time'])
        # out_path = Path(__file__).parents[0] / "evaluation_results" / "result_RL_OFDM.csv"
        # out_path.parent.mkdir(parents=True, exist_ok=True)
        # df.to_csv(out_path, index=False)

        total_time = time.perf_counter() - start
        return best_target, total_time, best_solution

    def load_model(self):
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        self._ac.load_state_dict(torch.load(path / 'ac.pt', map_location=self._device))

if __name__ == '__main__':
    device = 'cuda:0'
    solver = RLSolverOFDM(gpu=True, device=device, max_iter=1000)
    solver.load_model()
    perf, elapsed = solver.solve_all_evaluation_networks()
    print(f"Final Performance: {perf}, Time: {elapsed:.2f}s")