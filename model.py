import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.init import xavier_uniform_, constant_
from torch.distributions.categorical import Categorical
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import global_mean_pool

class AC(nn.Module):
    def __init__(self, num_power_level, num_rb, num_beam, power_attn_num_level, model_params, device):
        super(AC, self).__init__()
        self._device = device
        self._num_power_level = num_power_level
        self._num_rb = num_rb
        self._num_beam = num_beam
        self._power_attn_num_level = power_attn_num_level
        self._d_model = model_params['d_model']
        self._n_head = model_params['n_head']
        self._dim_feedforward = model_params['dim_feedforward']
        self._num_layers = model_params['actor_num_layers']
        self._dropout = model_params['dropout']

        self.gnn1 = GraphTransformer(
            input_dim=(self._num_power_level + self._num_beam) * self._num_rb,
            embedding_dim=self._power_attn_num_level * self._num_rb * self._num_beam,
            num_layers=self._num_layers, d_model=self._d_model, n_head=self._n_head,
            edge_dim=self._power_attn_num_level * self._num_rb * self._num_beam,
            dim_feedforward=self._dim_feedforward, dropout=self._dropout,
            activation="relu", device=self._device
        )

        self.link_rb_head = Linear(self._d_model, num_rb, device=device)
        self._rb_embedding = nn.Embedding(num_rb, self._d_model, device=device)

        # Second GNN: refine after link-RB selection


        # Final prediction heads
        self.beam_head = nn.Sequential(
            nn.Linear(self._d_model * 3, self._d_model, device=device),
            nn.ReLU(),
            nn.Linear(self._d_model, num_beam, device=device)
        )
        self.power_head = nn.Sequential(
            nn.Linear(self._d_model * 3, self._d_model, device=device),
            nn.ReLU(),
            nn.Linear(self._d_model, num_power_level, device=device)
        )

        self.value_head = Linear(self._d_model, 1, device=device)

        self._reset_parameters()

    def _reset_parameters(self):
        # nn.init.xavier_uniform_(self._actor_linear.weight)
        # nn.init.constant_(self._actor_linear.bias, 0.)
        # nn.init.xavier_uniform_(self._critic_linear.weight)
        # nn.init.constant_(self._critic_linear.bias, 0.)
        pass

    def forward(self, power_alloc, beam_alloc, node_power_attn, edge_power_attn, edge_index, ptr, batch):
        resource_alloc = torch.cat([power_alloc, beam_alloc], dim=2).reshape(power_alloc.size(0), -1)
        node_power_attn = node_power_attn.reshape(node_power_attn.size(0), -1)
        edge_power_attn = edge_power_attn.reshape(edge_power_attn.size(0), -1)
        x = self.gnn1(input=resource_alloc, node_embedding=node_power_attn,
                                    edge_attr=edge_power_attn, edge_index=edge_index)
        
        # link-RB 선택
        link_rb_logits = self.link_rb_head(x)  # [link, rb]
        link_rb_mask = (power_alloc.sum(dim=2) == 0)
        masked_logits = link_rb_logits.masked_fill(~link_rb_mask, -float('inf'))
        actions, log_probs, values, entropies = [], [], [], []
        global_ctx = x.mean(dim=0)

        for i in range(len(ptr) - 1):
            x_i = x[ptr[i]:ptr[i+1]]
            logits_i = masked_logits[ptr[i]:ptr[i+1]].flatten()
            if torch.all(torch.isinf(logits_i)):
                # fallback: return stop action (-1, -1, 0, 0) with 0 log_prob, value
                actions.append(torch.tensor([-1, -1, 0, 0], device=self._device))
                log_probs.append(torch.tensor(-torch.inf).to(self._device))
                values.append(torch.tensor(0.0, device=self._device))
                entropies.append(torch.tensor(0.0, device=self._device))
                continue
            dist_i = Categorical(logits=logits_i)
            idx = int(dist_i.sample())
            link = idx // self._num_rb
            rb = idx % self._num_rb
            logp_linkrb = dist_i.log_prob(torch.tensor(idx, device=self._device))

            # 선택된 정보
            x_selected = x_i[link]
            rb_emb = self._rb_embedding(torch.tensor(rb, device=self._device))
            ctx = torch.cat([x_selected, rb_emb, global_ctx], dim=-1)

            # beam/power logits from MLP
            beam_logits = self.beam_head(ctx)
            power_logits = self.power_head(ctx)

            beam_dist = Categorical(logits=beam_logits)
            power_dist = Categorical(logits=power_logits)

            beam = beam_dist.sample()
            power = power_dist.sample()

            logp_beam = beam_dist.log_prob(beam)
            logp_power = power_dist.log_prob(power)

            total_logp = logp_linkrb + logp_beam + logp_power

            actions.append(torch.stack([torch.tensor(link, device=self._device), torch.tensor(rb, device=self._device), beam, power]))
            log_probs.append(total_logp)
            values.append(self.value_head(x_i.mean(dim=0)).squeeze())
            entropies.append(beam_dist.entropy() + power_dist.entropy())

        actions = torch.stack(actions).to(self._device)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        entropies = torch.stack(entropies)

        return actions, log_probs, values, entropies

class ActDist:
    def __init__(self, logit, ptr, device):
        self._device = device
        self._ptr = ptr
        self._batch_size = int(ptr.shape[0]) - 1
        self._num_rb = logit.size(1)
        self._num_beam = logit.size(2)
        self._num_power_level = logit.size(3)
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
                node = idx // (self._num_rb * self._num_beam * self._num_power_level)
                rbbp = idx % (self._num_rb * self._num_beam * self._num_power_level)
                rb = rbbp // (self._num_beam * self._num_power_level)
                bp = rbbp % (self._num_beam * self._num_power_level)
                beam = bp // self._num_power_level
                power = bp % self._num_power_level
            else:
                node, rb, beam, power = -1, -1, -1, -1
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
                idx = node * self._num_rb * self._num_beam * self._num_power_level + rb * self._num_beam * self._num_power_level + beam * self._num_power_level + power
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