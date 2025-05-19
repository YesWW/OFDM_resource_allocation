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
from torch_geometric.utils import degree
from ofdm_simulator.ofdm_simulator_4 import OFDMSimulator
from model import AC
from utility import Buffer, get_buffer_dataloader
import wandb
import time
from torch_scatter import scatter_add

class trainer:
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
            self._gamma = self._config['train.gamma']
            self._lambda = self._config['train.lambda']
            self._buffer_batch_size = self._config['train.buffer_batch_size']
            self._ac_lr = self._config['train.ac_lr']

            self._rollout_batch_size = self._config['train.rollout_batch_size']
            self._min_attn_db = self._config['train.min_attn_db']
            self._max_attn_db = self._config['train.max_attn_db']
            self._power_attn_n_cls = self._config['train.power_attn.num_level']
            self._power_attn_boundaries = torch.linspace(self._min_attn_db, self._max_attn_db, self._power_attn_n_cls).to(self._device)
            
            # model architecture
            self._ac = AC(num_power=self._num_power_level, num_rb=self._num_rb, num_beam=self._num_beam,
                           power_attn_num_level=self._power_attn_n_cls, model_params=model_params, device=device)

            self._num_graph_repeat = self._config['train.num_graph_repeat']
            self._gamma = self._config['train.gamma']
            self._lambda = self._config['train.lambda']
            self._buffer_batch_size = self._config['train.buffer_batch_size']
            self._ac_lr = self._config['train.ac_lr']
            self._clip_max_norm = self._config['train.clip_max_norm']
            self._entropy_loss_weight = self._config['train.entropy_loss_weight']
            self._value_loss_weight =  self._config['train.value_loss_weight']

            self._PPO_clip = torch.Tensor([self._config['train.PPO_clip']]).to(torch.float).to(self._device)
            self._act_prob_ratio_exponent_clip = self._config['train.act_prob_ratio_exponent_clip']
            self._eval_period = self._config['train.eval_period']    
            self._eval_batch_size = self._config['train.eval_batch_size']

            self._eval_file_list = self._sim.get_evaluation_networks_file_list()
            self._num_evaluation_networks = len(self._eval_file_list)
            self._eval_networks = self._sim.generate_evaluation_networks(self._num_evaluation_networks)
            self._eval_data = self._sim.generate_pyg(self._eval_networks, min_attn_db=self._min_attn_db,
                                                      max_attn_db=self._max_attn_db, device=self._device)

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


    def roll_out(self, g, networks):
        num_node = g.x.size(0) 
        ptr, batch = g.ptr, g.batch
        batch_size = int(ptr.shape[0]) - 1
        power_alloc = torch.zeros(size=(num_node, self._num_rb, self._num_power_level)).to(self._device)
        beam_alloc = torch.zeros(size=(num_node, self._num_rb, self._num_beam)).to(self._device)
        link_rb_idx = self.get_order(g) # [batch, link*rb, 2]
        g2 = self.quantize_power_attn(g)
        # power_alloc[:,:,0] = 1
        # beam_alloc[:,:,0] = 1
        unterminated_node = torch.full(size=(num_node, self._num_rb), fill_value=True).to(self._device)
        ongoing = torch.full(size=(batch_size,), fill_value=True).to(self._device)

        power_alloc_buf = [] 
        beam_alloc_buf = []
        action_buf = []
        act_log_prob_buf = []
        value_buf = []
        ongoing_buf = []
        target_buf = []
        link_rb_buf = []
        self._ac.eval()

        with torch.no_grad():
            for i in range(link_rb_idx.size(1)):
            #while torch.any(ongoing):
                link_rb = link_rb_idx[:,i,:]
                action, act_log_prob, _, value = self._ac(power_alloc=power_alloc, beam_alloc=beam_alloc,
                                           node_power_attn=g2.x, edge_power_attn=g2.edge_attr, edge_index=g2.edge_index, ptr=ptr, batch=batch, link_rb=link_rb)
                # action = act_dist.sample()
                # act_log_prob = act_dist.log_prob(action)

                power_alloc_buf.append(power_alloc.detach().clone().cpu())
                beam_alloc_buf.append(beam_alloc.detach().clone().cpu())
                action_buf.append(action.detach().clone().cpu())
                act_log_prob_buf.append(act_log_prob.detach().clone().cpu())
                value_buf.append(value.detach().clone().cpu())
                ongoing_buf.append(ongoing.detach().clone().cpu())
                link_rb_buf.append(link_rb.detach().clone().cpu())

                target = []
                # update resource allocation
                for idx, act in enumerate(action):

                    power, beam = act//self._num_beam, act%self._num_beam
                    ptr_link = ptr[idx] + link_rb[idx][0]
                    rb = link_rb[idx][1]

                    # link_power_level = torch.nonzero(power_alloc[ptr_link][rb])
                    # power_alloc[ptr_link][rb][link_power_level] = 0
                    # power_alloc[ptr_link][rb][link_power_level+1] = 1
                    power_alloc[ptr_link][rb][power] = 1
                    
                    # link_beam_index = torch.nonzero(beam_alloc[ptr_link][rb])
                    # if link_beam_index.numel() != 0:
                    #     beam_alloc[ptr_link][rb][link_beam_index] = 0
                    beam_alloc[ptr_link][rb][beam] = 1

                    power_solution = torch.argmax(power_alloc[ptr[idx]:ptr[idx+1],:], dim=-1).cpu().numpy()
                    beam_solution = torch.argmax(beam_alloc[ptr[idx]:ptr[idx+1],:], dim=-1).cpu().numpy()
                    solution = {'power_level' : power_solution, 'beam_index' : beam_solution}
                    is_feasible = self._sim.is_solution_feasible(networks[idx], solution)
                    target_score = float(self._sim.get_optimization_target(networks[idx], solution))
                    val = torch.tensor(bool(is_feasible), dtype=torch.bool, device=self._device)
                    ongoing[idx] = val

                    target.append(target_score)
                target_buf.append(torch.Tensor(target))
        # store all of the interactions
        power_alloc_buf = torch.stack(power_alloc_buf, dim=0)
        beam_alloc_buf = torch.stack(beam_alloc_buf, dim=0)
        action_buf = torch.stack(action_buf, dim=0)
        act_log_prob_buf = torch.stack(act_log_prob_buf, dim=0)
        value_buf = torch.stack(value_buf, dim=0)
        ongoing_buf = torch.stack(ongoing_buf, dim=0)       
        target_buf = torch.stack(target_buf, dim=0)
        link_rb_buf = torch.stack(link_rb_buf, dim=0)

        buf = Buffer(g2, power_alloc_buf, beam_alloc_buf, action_buf, act_log_prob_buf, value_buf, ongoing_buf, target_buf, link_rb_buf, device=self._device)
        return buf


    def train(self, use_wandb=False, save_model=True):
        # initialize the wandb
        if use_wandb:
            wandb.init(project='OFDM_resource_allocation', config=self._config)
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("evaluate/*")
            if not save_model:
                wandb.watch((self._ac), log="all")
        train_step = 0
        train_start_time = time.perf_counter() 

        ac_param_dicts = [{"params": [p for n, p in self._ac.named_parameters() if p.requires_grad]}]
        ac_optimizer = torch.optim.Adam(ac_param_dicts, lr=self._ac_lr)

        for repeat_idx in range(self._num_graph_repeat):
            #num_ue = np.random.randint(low=self._config['sim.num_ue_range'][0], high=self._config['sim.num_ue_range'][1])
            networks = self._sim.get_networks(num_networks=self._rollout_batch_size) 
            g = self._sim.generate_pyg(networks=networks, min_attn_db=self._min_attn_db, max_attn_db=self._max_attn_db, device=self._device)
            buf = self.roll_out(g, networks)
            buf.cal_reward()
            buf.cal_lambda_return(gamma=self._gamma, lamb=self._lambda)
            buffer_dataloader = get_buffer_dataloader(buf, batch_size=self._buffer_batch_size, shuffle=True)
            self._ac.train()
            for minibatch_idx, d in enumerate(buffer_dataloader):
                torch.cuda.empty_cache()
                g = d['graph']
                power_alloc = d['power_alloc']
                beam_alloc = d['beam_alloc']
                action = d['action']
                link_rb = d['link_rb']
                init_act_log_prob = d['act_log_prob']
                lambda_return = d['return']                    
                advantage = lambda_return - d['value']
            
                # Train actor
                _, act_log_prob, entropy, value = self._ac(power_alloc=power_alloc, beam_alloc=beam_alloc, 
                               node_power_attn=g['x'], edge_power_attn=g['edge_attr'], 
                               edge_index=g['edge_index'], ptr=g['ptr'], batch=g['batch'], link_rb=link_rb)
                # Calculate PPO actor loss
                act_prob_ratio = torch.exp(torch.clamp(act_log_prob - init_act_log_prob,
                                                        max=self._act_prob_ratio_exponent_clip))
                actor_loss = torch.where(advantage >= 0,
                                            torch.minimum(act_prob_ratio, 1 + self._PPO_clip),
                                            torch.maximum(act_prob_ratio, 1 - self._PPO_clip))
                actor_loss = -(actor_loss * advantage)
                actor_loss = torch.mean(actor_loss)
                
                entropy_loss = -torch.mean(entropy)

                value_loss = nn.MSELoss()(value, lambda_return)

                total_loss = actor_loss + self._entropy_loss_weight * entropy_loss + self._value_loss_weight * value_loss
                ac_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._ac.parameters(), self._clip_max_norm)
                ac_optimizer.step()
                cumulative_time = time.perf_counter() - train_start_time
                train_step += 1
                # logging
                log = (f"repeat: {repeat_idx+1}/{self._num_graph_repeat}, "
                        f"minibatch: {minibatch_idx+1}/{len(buffer_dataloader)}, "
                        f"actor loss: {actor_loss}, value loss: {value_loss}, "
                        f"entropy loss: {entropy_loss}, total loss: {total_loss}")
                print(log)
                if use_wandb:
                    wandb_log = {"train/step": train_step, "train/actor_loss": actor_loss,
                                    "train/value_loss": value_loss, "train/entropy_loss": entropy_loss,
                                    "train/total_loss": total_loss, "train/time": cumulative_time}
                    wandb.log(wandb_log)

            torch.cuda.empty_cache()
            
            # graph data visualization
            if repeat_idx % self._eval_period == 0:
                self.evaluate(use_wandb)
                if save_model:
                    self.save_model(use_wandb)


    def evaluate(self, use_wandb=False):
        '''Evaluate the training data

        Args:
            use_wandb (bool): track the training results on wandb

        '''

        # eval data
        # networks = self._sim.get_networks(num_networks=self._eval_batch_size)
        # g = self._sim.generate_pyg(networks=networks, min_attn_db=self._min_attn_db, max_attn_db=self._max_attn_db, device=self._device)
        g = self._eval_data
        networks = self._eval_networks
        buf = self.roll_out(g, networks)
        buf.cal_reward()
        reward = buf.get_performance()
        target = buf._target.mean()

        reward = reward.mean()
        

        print(f"target value: {target}")

        if use_wandb:

            log = {"evaluate/reward": reward,
                   "evaluate/target": target}

            wandb.log(log)
            #wandb.log({'cir': wandb.Image(fig)})


    def save_model(self, use_wandb=False):
        '''Save the actor_critic weights

        Args:
            use_wandb (bool): save it on wandb platform
 
        '''
        if use_wandb:
            path = Path(wandb.run.dir)
        else:
            path = Path(__file__).parents[0].resolve() / 'saved_model'
        torch.save(self._ac.state_dict(), path / 'ac.pt')


    def load_model(self):
        '''Load the actor_critic weights

        '''
        
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        self._ac.load_state_dict(torch.load(path / 'ac.pt')) 


    def get_order(self, g):
        
        batch_size = len(g.ptr) - 1
        num_link = g.x.size(0) // batch_size

        #ptr = g.ptr

        sum_interf = scatter_add(g.edge_attr, g.edge_index[0].long(), dim=0).view(-1, self._num_rb, self._num_beam).mean(dim=2)
        # _, link_rb_index = sum_interf.flatten().sort()
        #link_rb_index = [sum_interf[ptr[i]:ptr[i+1]].flatten().sort()[1] for i in range(len(ptr)-1)]
        sorted_index = sum_interf.view(batch_size, -1, self._num_rb).view(batch_size, -1).sort(descending=True, dim=1)[1]
        
        link_idx = sorted_index // self._num_rb
        rb_idx   = sorted_index % self._num_rb
        link_rb_idx = torch.stack([link_idx, rb_idx], dim=2)
        
        return link_rb_idx


if __name__ == '__main__':
    device = 'cuda:0'
    tn = trainer(params_file='config.yaml', device=device)
    tn.train(use_wandb=False, save_model=True)
    #tn.evaluate()
    print(1)