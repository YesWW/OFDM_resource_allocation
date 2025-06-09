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
from ofdm_simulator.ofdm_simulator_rb import OFDMSimulator
from model import AC
from utility import Buffer, get_buffer_dataloader, ExpertStepDataset, get_step_dataloader
import wandb
import time

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

            
            
            # train parameter
            self._rollout_batch_size = self._config['train.rollout_batch_size']
            self._min_attn_db = self._config['train.min_attn_db']
            self._max_attn_db = self._config['train.max_attn_db']
            self._power_attn_n_cls = self._config['train.power_attn.num_level']
            self._power_attn_boundaries = torch.linspace(self._min_attn_db, self._max_attn_db, self._power_attn_n_cls).to(self._device)
            
            # model architecture
            self._ac = AC(num_power_level=self._num_power_level, num_rb=self._num_rb, num_beam=self._num_beam,
                           power_attn_num_level=self._power_attn_n_cls, model_params=model_params, device=device)

            self._num_graph_repeat = self._config['train.num_graph_repeat']
            self._gamma = self._config['train.gamma']
            self._lambda = self._config['train.lambda']
            self._buffer_batch_size = self._config['train.buffer_batch_size']
            self._ac_lr = self._config['train.ac_lr']
            self._clip_max_norm = self._config['train.clip_max_norm']
            self._entropy_loss_weight = self._config['train.entropy_loss_weight']
            self._value_loss_weight =  self._config['train.value_loss_weight']

            self._PPO_clip = torch.tensor(self._config['train.PPO_clip'], dtype=torch.float, device=self._device)
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

        g2 = self.quantize_power_attn(g)
        unallocated_rb = torch.full(size=(num_node, self._num_rb), fill_value=True).to(self._device)
        ongoing = torch.full(size=(batch_size,), fill_value=True).to(self._device)
        
        power_alloc_buf = [] 
        beam_alloc_buf = []
        action_buf = []
        act_log_prob_buf = []
        value_buf = []
        ongoing_buf = []
        target_buf = []
        self._ac.eval()

        with torch.no_grad():
            while torch.any(ongoing):
                logit_mask = self.get_masking(networks, power_alloc, batch)
                act_dist, value = self._ac(power_alloc=power_alloc, beam_alloc=beam_alloc, node_power_attn=g2.x, edge_power_attn=g2.edge_attr,
                                          edge_index=g2.edge_index, ptr=ptr, batch=batch, logit_mask=logit_mask)
                action = act_dist.sample()
                act_log_prob = act_dist.log_prob(action)

                power_alloc_buf.append(power_alloc.detach().clone().cpu())
                beam_alloc_buf.append(beam_alloc.detach().clone().cpu())
                action_buf.append(action.detach().clone().cpu())
                act_log_prob_buf.append(act_log_prob.detach().clone().cpu())
                value_buf.append(value.detach().clone().cpu())
                ongoing_buf.append(ongoing.detach().clone().cpu())

                target = []
                # update resource allocation
                for idx, act in enumerate(action):
                    if ongoing[idx]:
                        link, rb, power, beam = act
                        if link == -1 and rb == -1 and power == -1 and beam == -1:
                            ongoing[idx] = False
                            target_score = 0
                        else:

                            ptr_link = ptr[idx] + link

                            power_alloc[ptr_link][rb][power] = 1.0

                            beam_alloc[ptr_link][rb][beam] = 1.0

                            unallocated_rb[ptr_link][rb] = False

                            ongoing_alloc = torch.any(unallocated_rb[ptr[idx]: ptr[idx+1]])
                            power_solution = torch.argmax(power_alloc[ptr[idx]:ptr[idx+1],:], dim=-1).cpu().numpy()
                            beam_solution = torch.argmax(beam_alloc[ptr[idx]:ptr[idx+1],:], dim=-1).cpu().numpy()
                            solution = {'power_level' : power_solution, 'beam_index' : beam_solution}
                            target_score = float(self._sim.get_optimization_target(networks[idx], solution))
                            ongoing[idx] = ongoing_alloc
                    else:
                        target_score = 0
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

        buf = Buffer(g2, power_alloc_buf, beam_alloc_buf, action_buf, act_log_prob_buf, value_buf, ongoing_buf, target_buf, device=self._device)
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
                init_act_log_prob = d['act_log_prob']
                lambda_return = d['return']                    
                advantage = lambda_return - d['value']

            
                # Train actor
                act_dist, value = self._ac(power_alloc=power_alloc, beam_alloc=beam_alloc, 
                               node_power_attn=g['x'], edge_power_attn=g['edge_attr'], 
                               edge_index=g['edge_index'], ptr=g['ptr'], batch=g['batch'])
                # Calculate PPO actor loss
                act_log_prob = act_dist.log_prob(action)
                act_prob_ratio = torch.exp(torch.clamp(act_log_prob - init_act_log_prob,
                                                        max=self._act_prob_ratio_exponent_clip))
                actor_loss = torch.where(advantage >= 0,
                                            torch.minimum(act_prob_ratio, 1 + self._PPO_clip),
                                            torch.maximum(act_prob_ratio, 1 - self._PPO_clip))
                actor_loss = -(actor_loss * advantage)
                actor_loss = torch.mean(actor_loss)

                entropy_loss = -torch.mean(act_dist.entropy())

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


    def save_model(self, save_dir=None, filename='ac.pt', use_wandb=False):
        '''Save the actor_critic weights'''
        if save_dir is None:
            if use_wandb:
                path = Path(wandb.run.dir)
            else:
                path = Path(__file__).parents[0].resolve() / 'saved_model'
        else:
            path = Path(save_dir)

        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._ac.state_dict(), path / filename)


    def load_model(self):
        '''Load the actor_critic weights

        '''
        
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        self._ac.load_state_dict(torch.load(path / 'ac.pt')) 


    def get_masking(self, networks, power_alloc, batch):
        tx_power = torch.tensor(self._sim.tx_power, device=self._device)
        max_bs_power = self._sim.max_bs_power
        assoc = torch.cat([torch.tensor(network['assoc'], device=self._device) for network in networks], dim=0)

        node_power = (power_alloc * tx_power.view(1, 1, -1)).sum(dim=(1, 2))  # [num_node]

        bs_power = torch.zeros(batch.max() + 1, self._num_bs, device=self._device)  # [batch_size, num_bs]
        bs_idx = batch * self._num_bs + assoc  # [num_node] 
        bs_power_flat = bs_power.view(-1)  # [(batch_size * num_bs)]
        bs_power_flat = bs_power_flat.index_add(0, bs_idx, node_power.float()) 
        bs_power = bs_power_flat.view(-1, self._num_bs)  

        projected_power = bs_power[batch, assoc].view(-1, 1, 1, 1) + tx_power.view(1, 1, -1, 1)
        valid_mask = projected_power <= max_bs_power  # [num_node, num_rb, num_power_level, num_beam]
        valid_mask[:, :, 0, :] = True

        unterminated_mask = (torch.sum(power_alloc, dim=2) < 1.0)[:, :, None, None]  # [num_node, num_rb, 1, 1]
        logit_mask = valid_mask & unterminated_mask  
        return logit_mask.squeeze(-1)
    

    def load_expert_data(self, file_path, shuffle_step_order=True):
        expert_data = torch.load(file_path)
        networks = [item['network'] for item in expert_data]
        solutions = [item['solution'] for item in expert_data]

        g_batch = self._sim.generate_pyg(networks, min_attn_db=self._min_attn_db,
                                        max_attn_db=self._max_attn_db, device=self._device)
        g_batch = self.quantize_power_attn(g_batch)
        data_list = g_batch.to_data_list()

        all_steps = []

        for i, (net, sol) in enumerate(zip(networks, solutions)):
            graph = data_list[i]
            power_level = torch.tensor(sol['power_level'], dtype=torch.long)
            beam_index = torch.tensor(sol['beam_index'], dtype=torch.long)
            num_link, num_rb = power_level.shape

            step_indices = [(l, r) for l in range(num_link) for r in range(num_rb)]
            if shuffle_step_order:
                np.random.shuffle(step_indices)

            power_alloc = torch.zeros((num_link, num_rb, self._num_power_level))
            beam_alloc = torch.zeros((num_link, num_rb, self._num_beam))

            for link, rb in step_indices:
                state_power = power_alloc.clone()
                state_beam = beam_alloc.clone()

                p = power_level[link, rb].item()
                b = beam_index[link, rb].item()
                target = p * self._num_beam + b

                all_steps.append({
                    'graph': graph,
                    'power_alloc': state_power,
                    'beam_alloc': state_beam,
                    'link': link,
                    'rb': rb,
                    'target': target
                })

                power_alloc[link, rb, p] = 1.0
                beam_alloc[link, rb, b] = 1.0

        return all_steps


    def train_bc(self, expert_data_path, num_epochs=10, bc_lr=1e-4, batch_size=16, use_wandb=False):
        '''
        Tabu Search expert data로 Behavior Cloning 학습을 수행

        Args:
            expert_data_path (str): torch 저장된 expert data 파일 경로
            num_epochs (int): 학습 epoch 수
            bc_lr (float): learning rate
            batch_size (int): 배치 크기
        '''

        print("Loading expert data...")
        step_list = self.load_expert_data(expert_data_path)
        dataloader = get_step_dataloader(step_list, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._ac.parameters(), lr=bc_lr)
        self._ac.train()
        if use_wandb:
            wandb.init(project="OFDM_resource_allocation_IL", config=self._config)
            wandb.define_metric("train/epoch")
            wandb.define_metric("train/*", step_metric="train/epoch")
            wandb.define_metric("evaluate/*", step_metric="train/epoch")
            
        print("Start Behavior Cloning training...")
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch in dataloader:
                g = batch['graph'].to(self._device)
                power_alloc = batch['power_alloc'].to(self._device)
                beam_alloc = batch['beam_alloc'].to(self._device)
                link = batch['link'].to(self._device)
                rb = batch['rb'].to(self._device)
                target = batch['target'].to(self._device)

                act_dist, _ = self._ac(
                    power_alloc=power_alloc,
                    beam_alloc=beam_alloc,
                    node_power_attn=g.x,
                    edge_power_attn=g.edge_attr,
                    edge_index=g.edge_index,
                    ptr=g.ptr,
                    batch=g.batch
                )  # logit: [N, P, B]
                logit = act_dist._logit
                # (link, rb) 위치의 logit만 추출
                graph_idx = torch.arange(link.size(0), device=link.device)
                link_global = g.ptr[graph_idx] + link
                selected_logits = logit[link_global]  # [B, P, B]
                selected_logits = selected_logits.view(link.size(0), -1)

                loss = F.cross_entropy(selected_logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"[Epoch {epoch+1}/{num_epochs}] BC Loss: {avg_loss:.6f}")
            

            if use_wandb:
                wandb.log({"train/epoch": epoch + 1, "train/bc_loss": avg_loss})
            
            self.evaluate_bc(use_wandb=use_wandb)
            self.save_model(filename=f'ac_epoch{epoch+1}.pt', use_wandb=use_wandb)


    def evaluate_bc(self, use_wandb=False):
        '''
        Behavior Cloning 정책으로 평가 네트워크에 대해 성능을 측정
        '''
        self._ac.eval()

        g = self._eval_data  # PyG batch
        networks = self._eval_networks  # List of network dicts

        # rollout은 현재 self._ac를 사용하므로 BC 학습된 policy 평가가 됨
        buf = self.roll_out(g, networks)

        buf.cal_reward()
        target = buf._target.mean().item()
        reward = buf.get_performance().mean().item()

        print(f"[BC Evaluation] Target: {target:.4f}, Reward: {reward:.4f}")

        if use_wandb:
            wandb.log({
                "evaluate/bc_target": target,
                "evaluate/bc_reward": reward
            })



if __name__ == '__main__':
    device = 'cuda:0'
    tn = trainer(params_file='config.yaml', device=device)
    #tn.train(use_wandb=False, save_model=True)
    #tn.evaluate()
    tn.train_bc(expert_data_path='./expert_data.pt', num_epochs=20, use_wandb=True)
    print(1)