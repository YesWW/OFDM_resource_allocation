import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import dill
from torch_geometric.data import Data, Batch
import cupy as cp


class OFDMSimulator:
    def __init__(self, data_dir, tx_power_range, max_bs_power, noise_spectral_density=-174.0, alpha=1.0, num_rb=4,
                 num_ue_range=None, gpu=False):
        self.data_dir = Path(__file__).parents[0].resolve() / data_dir
        self.file_list = [os.path.join(self.data_dir, f)
                          for f in os.listdir(self.data_dir) if f.endswith('.pkl') and f != 'background.pkl']
        background_file = self.data_dir / 'background.pkl'
        with open(background_file, 'rb') as f:
            self.background = dill.load(f)
        with open(self.file_list[0], 'rb') as f:
            sample = dill.load(f)
        self.num_bs, self.num_rb, self.num_beam = sample['ch'].shape
        self.num_rb = num_rb
        self.num_ue_range = num_ue_range
        self.tx_power = np.linspace(start=0.0, stop=tx_power_range['max_power'], num=tx_power_range['num_power_level'])
        self.num_power_level = tx_power_range['num_power_level']
        self.max_bs_power = max_bs_power
        rb_size = self.background['rb_size']  # number of subcarriers in one RB
        subcarrier_spacing = self.background['subcarrier_spacing']  # subcarrier spacing (Hz)
        noise_spectral_density = np.power(10.0, noise_spectral_density / 10.0) / 1000.0  # noise spectral density (W/Hz)
        self._noise_power = noise_spectral_density * subcarrier_spacing * rb_size  # noise poser (W)
        self._alpha = alpha
        self._gpu = gpu

    def get_networks(self, num_networks):
        num_ue = np.random.randint(low=self.num_ue_range[0], high=self.num_ue_range[1])
        networks = []
        for _ in range(num_networks):
            networks.append(self.get_network(num_ue))
        return networks

    def get_network(self, num_ue):
        #num_ue = np.random.randint(low=self.num_ue_range[0], high=self.num_ue_range[1])
        file_list = np.random.choice(self.file_list, size=num_ue, replace=False)
        ch, pos = [], []
        for file in file_list:
            with open(file, 'rb') as f:
                data = dill.load(f)
            if np.sum(data['ch']) == 0.0:
                continue
            ch.append(data['ch'])
            pos.append(data['pos'])
        ch = np.stack(ch, axis=0)  # ue, bs, rb, beam
        ###
        ch = ch[:,:,:self.num_rb,:]
        ###
        pos = np.stack(pos, axis=0)  # ue, pos
        pow = np.sum(np.sum(ch, axis=-1), axis=-1)  # ue, bs
        assoc = np.argmax(pow, axis=1)
        return {'ch': ch, 'pos': pos, 'assoc': assoc}

    def generate_evaluation_networks(self, num_networks):
        networks = self.get_networks(num_networks)
        eval_dir = Path(__file__).parents[0].resolve() / 'evaluation_networks'
        for count, network in enumerate(networks):
            file_name = eval_dir / (str(count) + '.pkl')
            with open(file_name, "wb") as f:
                dill.dump(network, f)
        return networks

    def get_evaluation_networks_file_list(self):
        eval_dir = Path(__file__).parents[0].resolve() / 'evaluation_networks'
        file_list = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith('.pkl')]
        file_list.sort()
        return file_list

    def get_evaluation_network(self, file_name):
        file = Path(__file__).parents[0].resolve() / 'evaluation_networks' / file_name
        with open(file, 'rb') as f:
            network = dill.load(f)
        return network

    def generate_pyg(self, networks, min_attn_db, max_attn_db, device='cpu'):
        graph_list = []
        for network in networks:
            ch = network['ch']  # ue, bs, rb, beam
            # Normalize and flatten channel
            ch = 10.0 * np.log10(ch)  # dB scale
            ch = np.clip(ch, min_attn_db, max_attn_db)
            #ch = (ch - min_attn_db) / (max_attn_db - min_attn_db)
            ch = np.reshape(ch, newshape=(ch.shape[0], ch.shape[1], -1)) # ue, bs, rb*beam
            assoc = network['assoc']  # ue
            num_link = ch.shape[0]
            node_attr = []
            edge_index = []
            edge_attr = []
            for l in range(num_link):
                node_attr.append(ch[l, assoc[l]])
                for i in range(num_link):
                    if i != l:
                        interf = ch[l, assoc[i]]
                        #valid = np.any(interf > 0)
                        #if valid:
                        edge_index.append(np.array((i, l)))
                        edge_attr.append(interf)
            node_attr = torch.tensor(np.stack(node_attr, axis=0), dtype=torch.float32, device=device)
            edge_index = torch.tensor(np.stack(edge_index, axis=1), dtype=torch.int32, device=device)
            edge_attr = torch.tensor(np.stack(edge_attr, axis=0), dtype=torch.float32, device=device)
            graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
            graph_list.append(graph)
        batch = Batch.from_data_list(graph_list)
        return batch #graph_list#

    def get_optimization_target(self, network, solution):
        if self._gpu:
            target = self.get_optimization_target_gpu(network, solution)
        else:
            target = self.get_optimization_target_cpu(network, solution)
        return target

    def get_optimization_target_cpu(self, network, solution):
        power_level = solution['power_level']
        tx_power = np.take_along_axis(arr=self.tx_power[np.newaxis, np.newaxis, :],
                                      indices=power_level[:, :, np.newaxis], axis=-1)[:, :, 0]  # ue, rb
        beam_index = solution['beam_index']  # ue, rb
        ch = network['ch']  # ue, bs, rb, beam
        assoc = network['assoc']  # ue
        ch = np.take_along_axis(arr=ch[:, np.newaxis, :, :, :],
                                indices=assoc[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis],
                                axis=2)[:, :, 0, :, :]  # target ue, source ue, rb, beam
        ch = np.take_along_axis(arr=ch, indices=beam_index[np.newaxis, :, :, np.newaxis], axis=3)[:, :, :, 0]  # target ue, source ue, rb
        rx_power = ch * tx_power[np.newaxis, :, :]  # target ue, source ue, rb
        signal_power = np.swapaxes(np.diagonal(rx_power, axis1=0, axis2=1), axis1=0, axis2=1)  # ue, rb
        interf_power = np.sum(rx_power, axis=1) - signal_power  # ue, rb
        sinr = signal_power / (interf_power + self._noise_power)  # ue, rb
        spec_eff = np.mean(np.log2(1 + sinr), axis=1)  # ue
        se = spec_eff + 1E-20  # for numerical stability
        if self._alpha == 1.0:
            target = np.sum(np.log(se))
        else:
            target = np.sum(np.power(se, 1 - self._alpha) / (1 - self._alpha))
        return target

    def get_optimization_target_gpu(self, network, solution):
        power_level = cp.array(solution['power_level'])
        tx_power = cp.take_along_axis(a=cp.array(self.tx_power[cp.newaxis, cp.newaxis, :]),
                                      indices=power_level[:, :, cp.newaxis], axis=-1)[:, :, 0]  # ue, rb
        beam_index = cp.array(solution['beam_index'])  # ue, rb
        ch = cp.array(network['ch'])  # ue, bs, rb, beam
        assoc = cp.array(network['assoc'])  # ue
        ch = cp.take_along_axis(a=ch[:, cp.newaxis, :, :, :],
                                indices=assoc[cp.newaxis, :, cp.newaxis, cp.newaxis, cp.newaxis],
                                axis=2)[:, :, 0, :, :]  # target ue, source ue, rb, beam
        ch = cp.take_along_axis(a=ch, indices=beam_index[cp.newaxis, :, :, cp.newaxis], axis=3)[:, :, :, 0]  # target ue, source ue, rb
        rx_power = ch * tx_power[cp.newaxis, :, :]  # target ue, source ue, rb
        signal_power = cp.swapaxes(cp.diagonal(rx_power, axis1=0, axis2=1), axis1=0, axis2=1)  # ue, rb
        interf_power = cp.sum(rx_power, axis=1) - signal_power  # ue, rb
        sinr = signal_power / (interf_power + self._noise_power)  # ue, rb
        spec_eff = cp.mean(cp.log2(1 + sinr), axis=1)  # ue
        se = spec_eff + 1E-20  # for numerical stability
        if self._alpha == 1.0:
            target = cp.sum(cp.log(se))
        else:
            target = cp.sum(cp.power(se, 1 - self._alpha) / (1 - self._alpha))
        target = target.get()
        return target

    def generate_random_solution(self, networks):
        solutions = []
        for network in networks:
            num_ue, _, num_rb, num_beam = network['ch'].shape
            num_power_level = self.num_power_level
            power_level = np.random.randint(low=0, high=num_power_level, size=(num_ue, num_rb))
            beam_index = np.random.randint(low=0, high=num_beam, size=(num_ue, num_rb))
            solution = {'power_level': power_level, 'beam_index': beam_index}
            solutions.append(solution)
        return solutions

    def is_solution_feasible(self, network, solution):
        power_level = solution['power_level']
        power = np.take_along_axis(arr=self.tx_power[np.newaxis, np.newaxis, :],
                                   indices=power_level[:, :, np.newaxis], axis=-1)[:, :, 0]  # ue, rb
        power = np.sum(power, axis=-1)  # ue
        bs_power = np.zeros((power.shape[0], self.num_bs))
        np.put_along_axis(arr=bs_power, indices=network['assoc'][:, np.newaxis], values=power[:, np.newaxis], axis=-1)
        bs_power = np.sum(bs_power, axis=0)  # bs
        feasible = bs_power <= self.max_bs_power
        feasible = np.all(feasible)
        return feasible

    def plot(self, network):
        z_max = 2000
        ue_position = network['pos']
        ue_assoc = network['assoc']
        mesh_faces = self.background['mesh_faces']
        mesh_vertices = self.background['mesh_vertices']
        bs_position = self.background['bs_position']
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_zlim(0, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        x, y, z = mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2]
        ax.plot_trisurf(x, y, z, color='cyan', alpha=0.7, triangles=mesh_faces)
        for bs_pos in bs_position:
            x, y, z = bs_pos[0], bs_pos[1], bs_pos[2]
            ax.scatter(x, y, z, s=60, color='red', alpha=1.0, depthshade=False, edgecolors='red')
        for ue_pos, bs_idx in zip(ue_position, ue_assoc):
            ux, uy, uz = ue_pos[0], ue_pos[1], 5
            ax.scatter(ux, uy, uz, s=30, color='orange', alpha=1.0, depthshade=False, edgecolors='orange')
            bs_pos = bs_position[bs_idx]
            bx, by, bz = bs_pos[0], bs_pos[1], bs_pos[2]
            ax.plot([bx, ux], [by, uy], [bz, uz], color='black', linewidth=2)
        ax.view_init(elev=90, azim=-90)
        ax.set_proj_type('ortho')
        plt.show()

if __name__ == '__main__':
    data_dir = 'myeongdong_arr_4_rb_16'
    num_ue_range = [20, 40]  # Minimum and maximum number of UEs (randomly selected within range)
    tx_power_range = {'max_power': 2, 'num_power_level': 4}#16}  # Power level quantization
    max_bs_power = 10  # Maximum base station tx power (watt)
    noise_spectral_density = -174.0  # Noise spectral density (dBm/Hz)
    alpha = 0.0  # coefficient for alpha-fairness function (0.0: sum, 1.0: proportional, inf: max-min)
    gpu = True
    net = OFDMSimulator(data_dir, tx_power_range, max_bs_power, noise_spectral_density, alpha, num_ue_range, gpu)

    # # Plot test
    # network = net.get_network()
    # net.plot(network)
    #
    # # PyG test
    #networks = net.get_networks(num_networks=1)
    #net.generate_pyg_dataset(num_networks=100, min_attn_db=-200, max_attn_db=-50, device='cuda:0', file_name_prefix='train')
    #g = net.generate_pyg(num_networks=100, min_attn_db=-200, max_attn_db=-50, device='cuda:0')
    # print(1)
    #
    # Calculating performance target test
    networks = net.get_networks(num_networks=16)
    solutions = net.generate_random_solution(networks=networks)
    for i in range(16):
        print(net.is_solution_feasible(networks[i], solutions[i]))
    # opt_tgt = net.get_optimization_target(networks[0], solutions[0])
    # feasible = net.is_solution_feasible(networks[0], solutions[0])

    # # Generate evaluation dataset
    # net.generate_evaluation_networks(10)
    # eval_file_list = net.get_evaluation_networks_file_list()
    # network = net.get_evaluation_network(eval_file_list[0])
    # net.plot(network)
    # pass





