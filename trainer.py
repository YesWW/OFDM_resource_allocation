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
from utility import Buffer, get_buffer_dataloader, ExpertDataset, get_expert_dataloader
import wandb
import time
from tabu_search import TabuSearch
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

            self._tabu = TabuSearch(
                self._sim.data_dir,
                self._config['sim.tx_power_range'],
                self._sim.max_bs_power,
                noise_spectral_density=-174.0,
                alpha=0.0,
                tabu_move_selection_prob=0.01,
                tabu_max_iterations=200,
                tabu_tenure=100,
                gpu=True)

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
        # unterminated_node = torch.full(size=(num_node, self._num_rb), fill_value=True).to(self._device)
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
                action, act_log_prob, _, value, _ = self._ac(power_alloc=power_alloc, beam_alloc=beam_alloc,
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
        torch.cuda.empty_cache()
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


    def train_bc(self, use_wandb=False, save_model=True):
        if use_wandb:
            wandb.init(project='OFDM_resource_allocation_IL', config=self._config)
            ...
            
        ac_optimizer = torch.optim.Adam(
            [p for p in self._ac.parameters() if p.requires_grad],
            lr=self._ac_lr
        )
        self._ac.train()
        train_step = 0
        il_epochs = self._config.get('train.il_epochs', 20)
        il_repeat = 1000  # 🔁 반복 횟수 설정

        for repeat_idx in range(il_repeat):
            print(f"\n=== IL Repeat {repeat_idx+1}/{il_repeat} ===")

            # 1. 샘플링 & 그래프 생성
            networks = self._sim.get_networks(num_networks=self._rollout_batch_size)
            g = self._sim.generate_pyg(networks, min_attn_db=self._min_attn_db,
                                    max_attn_db=self._max_attn_db, device=self._device)
            link_rb_idx = self.get_order(g)
            g2 = self.quantize_power_attn(g)
            
            # 2. expert solution 생성
            allocs = []
            for net in networks:
                sol = self._tabu.solve_bc(net)
                power = torch.tensor(sol['power_level'])
                beam  = torch.tensor(sol['beam_index'])
                joint = (power * self._num_beam + beam)
                allocs.append(joint)
            target_joint = torch.stack(allocs, dim=0).float().to(self._device)
            
            target_joint = F.one_hot(target_joint.view(target_joint.size(0), -1).long(), num_classes=self._num_power_level*self._num_beam).float()
            # 3. 초기 상태 설정
            num_nodes = g2.x.size(0)
            power_alloc = torch.zeros((num_nodes, self._num_rb, self._num_power_level), device=self._device)
            beam_alloc  = torch.zeros((num_nodes, self._num_rb, self._num_beam), device=self._device)
            # power_alloc[:, :, 0] = 1
            # beam_alloc[:, :, 0] = 1

            # 4. il_epochs 동안 반복 학습
            for epoch in range(il_epochs):
                
                torch.cuda.empty_cache()
                for i in range(link_rb_idx.size(1)):
                    link_rb = link_rb_idx[:,i,:]
                    link_rb_joint = link_rb[:,0] * self._num_beam + link_rb[:,1]
                    target_link_rb = target_joint[torch.arange(link_rb_joint.size(0), device=self._device),link_rb_joint]
                    _, _, _, _, logit = self._ac(
                        power_alloc, beam_alloc,
                        node_power_attn=g2.x, edge_power_attn=g2.edge_attr,
                        edge_index=g2.edge_index, ptr=g2.ptr, batch=g2.batch, link_rb=link_rb)

                loss = F.cross_entropy(logit, target_link_rb)

                ac_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._ac.parameters(), self._clip_max_norm)
                ac_optimizer.step()

                print(f"[Repeat {repeat_idx+1}, Epoch {epoch+1}] BC loss: {loss.item():.4f}")
                train_step += 1

                if use_wandb:
                    wandb.log({"train/step": train_step, "train/il_loss": loss.item()})

            # 반복 종료 시 평가/모델 저장
            if repeat_idx % self._eval_period == 0:
                self.evaluate(use_wandb)
                if save_model:
                    self.save_model(use_wandb)


    def get_order(self, g):
        
        batch_size = len(g.ptr) - 1
        num_link = g.x.size(0) // batch_size

        #ptr = g.ptr
        
        sum_interf = scatter_add(torch.argmax(g.edge_attr, dim=2).float(), g.edge_index[0].long(), dim=0).view(-1, self._num_rb, self._num_beam).mean(dim=2)
        # _, link_rb_index = sum_interf.flatten().sort()
        #link_rb_index = [sum_interf[ptr[i]:ptr[i+1]].flatten().sort()[1] for i in range(len(ptr)-1)]
        sorted_index = sum_interf.view(batch_size, -1, self._num_rb).view(batch_size, -1).sort(descending=True, dim=1)[1]
        
        link_idx = sorted_index // self._num_rb
        rb_idx   = sorted_index % self._num_rb
        link_rb_idx = torch.stack([link_idx, rb_idx], dim=2)
        
        return link_rb_idx

    def generate_expert_dataset(self, save_path='expert_dataset.pt', num_samples=1000):
        all_graphs = []
        all_targets = []
        for i in tqdm(range(num_samples // self._rollout_batch_size)):
            networks = self._sim.get_networks(num_networks=self._rollout_batch_size)
            g = self._sim.generate_pyg(networks, min_attn_db=self._min_attn_db,
                                    max_attn_db=self._max_attn_db, device=self._device)
            link_rb_idx = self.get_order(g)
            g2 = self.quantize_power_attn(g)

            allocs = []
            for net in networks:
                sol = self._tabu.solve_bc(net)
                power = torch.tensor(sol['power_level'])
                beam  = torch.tensor(sol['beam_index'])
                joint = (power * self._num_beam + beam)
                allocs.append(joint)

            joint_alloc = torch.stack(allocs, dim=0)  # [batch, link, rb]
            all_graphs.append(g2.to('cpu'))
            all_targets.append(joint_alloc.to('cpu'))

        dataset = {
            'graphs': all_graphs,
            'targets': all_targets
        }
        torch.save(dataset, save_path)
        print(f"Saved expert dataset to: {save_path}")

    def train_bc_from_dataset_batch(self, dataset_path='expert_dataset.pt', use_wandb=False, save_model=True):
        # Load and flatten
        dataset = torch.load(dataset_path)
        graph_batches = dataset['graphs']
        target_batches = dataset['targets']

        
        graph_list = []
        target_list = []
        power_list = []
        beam_list = []
        link_rb_list = []

        for g_batch, t_batch in zip(graph_batches, target_batches):
            link_rb_idx = self.get_order(g_batch)
            power_alloc = torch.zeros(size=(g_batch.x.size(0), self._num_rb, self._num_power_level)).to(self._device)
            beam_alloc = torch.zeros(size=(g_batch.x.size(0), self._num_rb, self._num_beam)).to(self._device)
            ptr = g_batch.ptr
            graph_list.append(g_batch)
            target_list.append(t_batch)
            p_list = []
            b_list = []
            lr_list = []
            for lr in range(link_rb_idx.size(1)):
                link_rb = link_rb_idx[:, lr, :]
                for idx in range(len(ptr)-1):
                    link, rb = link_rb[idx]
                    target = t_batch[idx][link][rb]
                    power = target // self._num_beam
                    beam = target % self._num_beam
                    ptr_link = ptr[idx]+link
                    power_alloc[ptr_link][rb][power] = 1.0
                    beam_alloc[ptr_link][rb][beam] = 1.0
                p_list.append(power_alloc)
                b_list.append(beam_alloc)
                lr_list.append(link_rb)
            power_list.append(torch.stack(p_list))
            beam_list.append(torch.stack(b_list))
            link_rb_list.append(torch.stack(lr_list))

        # Dataset & DataLoader
        expert_dataset = ExpertDataset(graph_list, target_list, power_list, beam_list, link_rb_list)
        dataloader = get_expert_dataloader(expert_dataset,  batch_size=self._buffer_batch_size, shuffle=True)

        ac_optimizer = torch.optim.Adam(
            [p for p in self._ac.parameters() if p.requires_grad],
            lr=self._ac_lr
        )
        self._ac.train()
        train_step = 0
        il_epochs = self._config.get('train.il_epochs', 20)

        for epoch in range(il_epochs):
            for batch_idx, b in enumerate(dataloader):
                g2 = b['graph'].to(self._device)
                target = b['target'].to(self._device)
                power_alloc = b['power_alloc'].to(self._device)
                beam_alloc = b['beam_alloc'].to(self._device)
                link_rb = b['link_rb'].to(self._device)
                target = F.one_hot(target.long(), num_classes=self._num_power_level * self._num_beam).to(self._device)
                total_loss = 0


                _, _, _, _, logit = self._ac(
                    power_alloc=power_alloc, beam_alloc=beam_alloc,
                    node_power_attn=g2.x, edge_power_attn=g2.edge_attr,
                    edge_index=g2.edge_index, ptr=g2.ptr, batch=g2.batch, link_rb=link_rb)

                loss = F.cross_entropy(logit, target.float())
                ac_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._ac.parameters(), self._clip_max_norm)
                ac_optimizer.step()

                total_loss += loss.item()
                train_step += 1
                if use_wandb:
                    wandb.log({"train/step": train_step, "train/il_loss": loss.item()})

                print(f"[Epoch {epoch+1} | Batch {batch_idx+1}] Total IL loss: {total_loss:.4f}")

            # periodic evaluation
            if epoch % self._eval_period == 0:
                self.evaluate(use_wandb)
                if save_model:
                    self.save_model(use_wandb)







if __name__ == '__main__':
    device = 'cuda:0'
    tn = trainer(params_file='config.yaml', device=device)
    #tn.load_model()
    #tn.train(use_wandb=False, save_model=True)
    #tn.evaluate()
    #tn.train_bc(use_wandb=False, save_model=True)
    #tn.load_model()
    #tn.train_bc(use_wandb=True, save_model=True)
    #tn.train(use_wandb=False, save_model=True)
    #tn.generate_expert_dataset()
    tn.train_bc_from_dataset_batch(use_wandb=True, save_model=True)
    print(1)