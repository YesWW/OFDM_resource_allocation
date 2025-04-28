import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.init import xavier_uniform_, constant_
from torch.distributions.categorical import Categorical
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.data import Batch

class AC(nn.Module):
    def __init__(self, num_max_link, num_power_level, num_rb, num_beam, power_attn_num_level, model_params, device):
        super(AC, self).__init__()
        self._device = device
        self._num_max_link = num_max_link
        self._num_power_level = num_power_level
        self._num_rb = num_rb
        self._num_beam = num_beam
        self._power_attn_num_level = power_attn_num_level
        self._d_model = model_params['d_model']
        self._n_head = model_params['n_head']
        self._dim_feedforward = model_params['dim_feedforward']
        self._num_layers = model_params['actor_num_layers']
        self._dropout = model_params['dropout']
        self._rb_emb_dim = 128  #  추가: RB embedding 차원

        self._graph_transformer = GraphTransformer(
            input_dim=(num_power_level + num_beam) * num_rb,
            embedding_dim=power_attn_num_level * num_rb * num_beam,
            num_layers=self._num_layers, d_model=self._d_model, n_head=self._n_head,
            edge_dim=power_attn_num_level * num_rb * num_beam,
            dim_feedforward=self._dim_feedforward, dropout=self._dropout,
            activation="relu", device=self._device
        )

        self._actor_linear = Linear(self._d_model, num_rb, device=device)
        self._low_actor_linear = Linear(self._d_model, num_beam, device=device)
        self._critic_linear = Linear(self._d_model, 1, device=device)
        self._rb_embedding = nn.Embedding(num_embeddings=num_rb, embedding_dim=self._rb_emb_dim, device=device)
        self._beam_mlp = nn.Sequential(
            nn.Linear(self._d_model + self._rb_emb_dim, self._d_model, device=device),
            nn.ReLU(),
            nn.Linear(self._d_model, num_beam, device=device)
        )
        # self._link_rb_head = nn.Linear(self._d_model, num_rb, device=device)
        # self._power_beam_head = nn.Linear(self._d_model, num_power_level*num_beam, device=device)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self._actor_linear.weight)
        nn.init.constant_(self._actor_linear.bias, 0.)
        nn.init.xavier_uniform_(self._critic_linear.weight)
        nn.init.constant_(self._critic_linear.bias, 0.)
        nn.init.xavier_uniform_(self._low_actor_linear.weight)
        nn.init.constant_(self._low_actor_linear.bias, 0.)

    def forward(self, power_alloc, beam_alloc, node_power_attn, edge_power_attn, edge_index, ptr, batch):
        resource_alloc = torch.cat([power_alloc, beam_alloc], dim=2).reshape(power_alloc.size(0), -1)
        node_power_attn = node_power_attn.reshape(node_power_attn.size(0), -1)
        edge_power_attn = edge_power_attn.reshape(edge_power_attn.size(0), -1)

        x = self._graph_transformer(input=resource_alloc, node_embedding=node_power_attn,
                                    edge_attr=edge_power_attn, edge_index=edge_index)
        
        value = global_mean_pool(x=x, batch=batch)
        value = self._critic_linear(value)[:, 0]

        

        action_list = []
        log_prob_list = []
        entropy_list = []
        high_entropy_list = []
        low_entropy_list = []
        link_rb_logit = self._actor_linear(x)   # x: [link, d_model] -> logit: [link, rb]

        for i in range((len(ptr)-1)):
            link_rb_logit_i = link_rb_logit[ptr[i]: ptr[i+1]].view(-1)  # [real_link, rb] -> [real_link*rb,]
            link_rb_dist = Categorical(logits=link_rb_logit_i)
            link_rb_sample = link_rb_dist.sample()
            link = link_rb_sample // self._num_rb
            rb = link_rb_sample % self._num_rb
            
            selected_x = x[ptr[i] + link]  # 선택된 link feature

            # Step 2: Power selection
            #power_input = torch.cat([x_i, rb_embed], dim=-1)
            rb_emb = self._rb_embedding(rb)
            beam_input = torch.cat([selected_x, rb_emb], dim=-1)

            beam_logits = self._beam_mlp(beam_input)
            beam_dist = Categorical(logits=beam_logits)
            beam_sample = beam_dist.sample()

            # Store per-graph results
            action = torch.stack([link, rb, beam_sample], dim=-1)
            log_prob = link_rb_dist.log_prob(link_rb_sample) + beam_dist.log_prob(beam_sample)
            entropy = link_rb_dist.entropy() + beam_dist.entropy()

            action_list.append(action)
            log_prob_list.append(log_prob)
            
            high_entropy_list.append(link_rb_dist.entropy())
            low_entropy_list.append(beam_dist.entropy())
            entropy_list.append(entropy)

        # Concatenate all graphs' results
        action_ = torch.stack(action_list, dim=0)
        log_prob_ = torch.stack(log_prob_list, dim=0)
        entropy_ = torch.stack(entropy_list, dim=0)
        high_entropy_ = torch.stack(high_entropy_list, dim=0)
        low_entropy_ = torch.stack(low_entropy_list, dim=0)

        return action_, log_prob_, entropy_, value, high_entropy_, low_entropy_

        # # (R, K, 3)
        # rb_ids = torch.arange(self._num_rb, device=self._device)        # (R,)
        # beam_ids = torch.arange(self._num_beam, device=self._device)    # (K,)
        # delta_ids = torch.arange(3, device=self._device)                # (3,)

        # # 2. Cartesian product: (R*K*3, 3)
        # action_idx = torch.cartesian_prod(rb_ids, beam_ids, delta_ids).to(self._device)

        # # 3. Split out for embedding
        # rb_flat, beam_flat, delta_flat = action_idx[:, 0], action_idx[:, 1], action_idx[:, 2]

        # # 4. Get embeddings: shape (num_actions, d)
        # rb_vecs = self.rb_emb(rb_flat)
        # beam_vecs = self.beam_emb(beam_flat)
        # delta_vecs = self.power_emb(delta_flat)

        # # 5. Concatenate: shape (num_actions, 3d)
        # action_embs = torch.cat([rb_vecs, beam_vecs, delta_vecs], dim=-1)

        # num_links = x.size(0)  # GNN 출력: x → [num_links, d_model]
        # num_actions = action_embs.size(0)  # = R * K * 3
        # # → (num_links, 1, d_model)
        # x_expanded = x.unsqueeze(1)

        # # → (1, num_actions, 3d)
        # a_expanded = action_embs.unsqueeze(0)

        # # → (num_links, num_actions, d_model + 3d)
        # joint = torch.cat([x_expanded.expand(-1, num_actions, -1), a_expanded.expand(num_links, -1, -1)], dim=-1)

        # # → (num_links, num_actions, 1)
        # logit = self.act_proj(joint)

        # # → reshape to (num_links, R, K, 3)
        # logit = logit.view(num_links, self._num_rb, self._num_beam, 3)

        # # if power == max: -inf
        # unterminated_mask = ~(power_alloc[:, :, -1] == 1)
        # logit = torch.where(condition=unterminated_mask.unsqueeze(2).unsqueeze(3), input=logit, other=-torch.inf)
        # # power == 0 -> dont assign power = 0
        # zero_power_mask = (power_alloc[:,:,0] == 1).unsqueeze(2)
        # logit[:, :, :, 0] = torch.where(zero_power_mask.expand(-1, -1, logit.size(2)), -torch.inf, logit[:, :, :, 0])

        action_list = []
        log_prob_list = []
        entropy_list = []

        for i in range((len(ptr)-1)):
            link_rb_logit_i = link_rb_logit[ptr[i]: ptr[i+1]].view(-1)  # [real_link, rb] -> [real_link*rb,]
            link_rb_dist = Categorical(logits=link_rb_logit_i)
            link_rb = link_rb_dist.sample()
            link = link_rb // self._num_rb
            rb = link_rb % self._num_rb
            
            selected_x = x[ptr[i]+link]   # [d_model]

            # Step 2: Power selection
            #power_input = torch.cat([x_i, rb_embed], dim=-1)
            beam_logits = self._low_actor_linear(selected_x)  # (num_beam,)
            beam_dist = Categorical(logits=beam_logits)
            beam_sample = beam_dist.sample()

            # Store per-graph results
            action = torch.stack([link, rb, beam_sample], dim=-1)  # (link, rb, beam)
            log_prob = link_rb_dist.log_prob(link_rb) + beam_dist.log_prob(beam_sample)
            entropy = link_rb_dist.entropy() + beam_dist.entropy()

            action_list.append(action)
            log_prob_list.append(log_prob)
            entropy_list.append(entropy)

        # Concatenate all graphs' results
        action_ = torch.stack(action_list, dim=0)
        log_prob_ = torch.stack(log_prob_list, dim=0)
        entropy_ = torch.stack(entropy_list, dim=0)

        return action_, log_prob_, entropy_, value
        
        # #logit[terminated_mask] = -torch.inf

        # act_dist = ActDist(logit, ptr, device=self._device)
        # return act_dist, value

class ActDist:
    def __init__(self, logit, ptr, device):
        self._device = device
        self._ptr = ptr
        self._batch_size = int(ptr.shape[0]) - 1
        self._num_rb = logit.size(1)
        self._num_beam = logit.size(2)
        self._power_fix_size = logit.size(3)
        self._dist_list = []
        for idx in range(self._batch_size):
            l = logit[ptr[idx]: ptr[idx + 1], :, :].to(self._device)
            l = torch.flatten(l)
            if torch.all(torch.isinf(l)):
                dist = None
            else:
                dist = Categorical(logits=l)
            self._dist_list.append(dist)

    def sample(self):
        action = []
        for dist in self._dist_list:
            if dist is not None:
                idx = int(dist.sample())
                node = idx // (self._num_rb * self._num_beam * self._power_fix_size)
                rb_remain = idx % (self._num_rb * self._num_beam * self._power_fix_size)
                rb = rb_remain // (self._num_beam * self._power_fix_size)
                beam_remain = rb_remain % (self._num_beam * self._power_fix_size)
                beam = beam_remain // self._power_fix_size
                power = beam_remain % self._power_fix_size
            else:
                node, rb, beam, power = -1, -1, -1, 0
            action.append([node, rb, beam, power])
        action = torch.Tensor(action).to(torch.int).to(self._device)
        return action
    
    def entropy(self):
        entropy = []
        for dist in self._dist_list:
            entropy.append(dist.entropy() if dist is not None else torch.tensor(0.0))
        entropy = torch.Tensor(entropy).to(self._device)
        return entropy

    def log_prob(self, action):
        action = torch.Tensor(action).to(self._device)
        lp = []
        for a, dist in zip(action, self._dist_list):
            if dist is not None:
                node, rb, beam, power = a
                idx = node * self._num_rb * self._num_beam * self._power_fix_size + rb * self._num_beam * self._power_fix_size + beam * self._power_fix_size + power
                lp.append(dist.log_prob(idx))
            else:
                lp.append(torch.tensor(-torch.inf).to(self._device))
        lp = torch.stack(lp, dim=0)
        return lp


class GraphTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_layers, d_model, n_head, edge_dim, dim_feedforward, dropout, activation="relu", device='cpu'):
        super(GraphTransformer, self).__init__()
        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._d_model = d_model
        self._n_head = n_head
        self._edge_dim = edge_dim
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._device = device
        self._activation = activation
        self._input_linear = Linear(in_features=self._input_dim, out_features=self._d_model, bias=True, device=device)
        self._node_embedding_linear = Linear(in_features=self._embedding_dim, out_features=self._d_model, bias=True, device=device)
        self._layer_list = nn.ModuleList()
        for _ in range(self._num_layers):
            layer = GraphTransformerLayer(d_model=self._d_model, n_head=self._n_head,
                                          edge_dim=self._edge_dim,
                                          dim_feedforward=self._dim_feedforward, dropout=self._dropout,
                                          activation=self._activation, device=self._device)
            self._layer_list.append(layer)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self._input_linear.weight)
        xavier_uniform_(self._node_embedding_linear.weight)
        constant_(self._input_linear.bias, 0.)
        constant_(self._node_embedding_linear.bias, 0.)

    def forward(self, input, node_embedding, edge_attr, edge_index):
        # input_ = self._input_linear(input) + self._node_embedding_linear(node_embedding)
        # x = torch.zeros_like(input_).to(self._device)
        input = self._input_linear(input)
        x = self._node_embedding_linear(node_embedding)
        for layer in self._layer_list:
            x = x + input
            x = layer(x, edge_attr, edge_index)
        return x


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, edge_dim, dim_feedforward, dropout, activation="relu", device='cpu'):
        super(GraphTransformerLayer, self).__init__()
        self._d_model = d_model
        self._n_head = n_head
        self._edge_dim = edge_dim
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._device = device
        self._activation = activation
        # Transformer convolution
        out_channel = d_model // n_head
        self._trans_conv = TransformerConv(in_channels=d_model, out_channels=out_channel, heads=n_head,
                                           concat=True, beta=False, dropout=dropout, edge_dim=edge_dim,
                                           bias=True, root_weight=True).to(device)
        # Feedforward neural network
        self.ffnn_linear1 = Linear(in_features=d_model, out_features=dim_feedforward, bias=True, device=device)
        self.ffnn_dropout = Dropout(dropout)
        self.ffnn_linear2 = Linear(in_features=dim_feedforward, out_features=d_model, bias=True, device=device)
        # Layer norm and dropout
        layer_norm_eps = 1e-5
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        # Activation
        self.activation = self._get_activation_fn(activation)
        # Reset parameters
        self._reset_parameters()

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")

    def _reset_parameters(self):
        xavier_uniform_(self.ffnn_linear1.weight)
        xavier_uniform_(self.ffnn_linear2.weight)
        constant_(self.ffnn_linear1.bias, 0.)
        constant_(self.ffnn_linear2.bias, 0.)
        self._trans_conv.reset_parameters()

    def forward(self, x, edge_attr, edge_index):
        x2 = self._trans_conv(x=x, edge_index=edge_index.long(), edge_attr=edge_attr, return_attention_weights=None)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.ffnn_linear2(self.ffnn_dropout(self.activation(self.ffnn_linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x