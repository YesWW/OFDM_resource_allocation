import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.init import xavier_uniform_, constant_
from torch.distributions.categorical import Categorical
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import to_dense_batch

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

        self._graph_transformer = GraphTransformer(
            input_dim=(self._num_power_level + self._num_beam) * self._num_rb,
            embedding_dim=self._power_attn_num_level * self._num_rb * self._num_beam,
            num_layers=self._num_layers, d_model=self._d_model, n_head=self._n_head,
            edge_dim=self._power_attn_num_level * self._num_rb * self._num_beam,
            dim_feedforward=self._dim_feedforward,
            dropout=self._dropout,
            activation="relu",
            device=self._device
        )
        self.link_head = Linear(self._d_model, 1, device=self._device)  # output single score per link
        self.rb_embedding = nn.Embedding(num_rb, self._d_model, device=self._device)
        self.pb_head = nn.Sequential(
            nn.Linear(self._d_model * 2, self._d_model, device=self._device),
            nn.ReLU(),
            nn.Linear(self._d_model, num_power_level * num_beam, device=self._device)
        )
        self.critic_head = Linear(self._d_model, 1, device=self._device)

    def forward(self, power_alloc, beam_alloc, node_power_attn, edge_power_attn, edge_index, ptr, batch):
        resource_alloc = torch.cat([power_alloc, beam_alloc], dim=2).reshape(power_alloc.size(0), -1)
        node_power_attn = node_power_attn.reshape(node_power_attn.size(0), -1)
        edge_power_attn = edge_power_attn.reshape(edge_power_attn.size(0), -1)

        h = self._graph_transformer(input=resource_alloc, node_embedding=node_power_attn,
                     edge_attr=edge_power_attn, edge_index=edge_index)
        h_padded, mask = to_dense_batch(h, batch)  # [B, N_max, d_model], [B, N_max]
        link_logits = self.link_head(h_padded).squeeze(-1)  # [B, N_max]
        link_logits = link_logits.masked_fill(~mask, float('-inf'))

        values = self.critic_head(global_mean_pool(h, batch))[:, 0]  # [batch_size]
        return h_padded, link_logits, values, mask

    def sample(self, h_padded, link_logits, rb_idx, mask):
        batch_size, max_nodes, d_model = h_padded.size()

        link_logits = torch.nan_to_num(link_logits, nan=-1e9)
        all_inf_mask = ~torch.isfinite(link_logits).any(dim=1)

        action = torch.full((batch_size, 4), -1, dtype=torch.long, device=self._device)
        total_log_prob = torch.full((batch_size,), -float('inf'), device=self._device)
        entropy = torch.zeros(batch_size, device=self._device)

        valid_idx = (~all_inf_mask).nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() > 0:
            mask_valid = mask[valid_idx]  # [V, N]
            logits_valid = link_logits[valid_idx]  # [V, N]
            h_valid = h_padded[valid_idx]  # [V, N, d_model]

            dist_link = Categorical(logits=logits_valid)
            local_node_idx = dist_link.sample()  # [V]

            # (1) 벡터화된 인덱스 변환
            flat_mask_idx = mask_valid.nonzero(as_tuple=False)  # [?, 2]
            valid_node_idx = flat_mask_idx[:, 1]
            counts = mask_valid.sum(dim=1)
            ptr = torch.cat([torch.tensor([0], device=self._device), counts.cumsum(dim=0)])
            global_node_idx = valid_node_idx[ptr[:-1] + local_node_idx]  # [V]

            # (2) PB prediction
            node_idx_exp = local_node_idx.view(-1, 1, 1).expand(-1, 1, d_model)
            h_selected = torch.gather(h_valid, dim=1, index=node_idx_exp).squeeze(1)
            rb_valid = rb_idx[valid_idx]
            rb_emb = self.rb_embedding(rb_valid)
            pb_input = torch.cat([h_selected, rb_emb], dim=-1)
            pb_logits = self.pb_head(pb_input)

            dist_pb = Categorical(logits=pb_logits)
            pb_sample = dist_pb.sample()
            power = pb_sample // self._num_beam
            beam = pb_sample % self._num_beam

            # (3) Build action
            action[valid_idx, 0] = global_node_idx
            action[valid_idx, 1] = rb_valid
            action[valid_idx, 2] = power
            action[valid_idx, 3] = beam

            total_log_prob[valid_idx] = dist_link.log_prob(local_node_idx) + dist_pb.log_prob(pb_sample)
            entropy[valid_idx] = dist_link.entropy() + dist_pb.entropy()

        return action, total_log_prob, entropy

    def log_prob(self, h_padded, link_logits, actions):
        link = actions[:, 0]
        rb_idx = actions[:, 1]
        power = actions[:, 2]
        beam = actions[:, 3]

        valid_mask = (link >= 0) & (rb_idx >= 0) & (power >= 0) & (beam >= 0)
        log_prob = torch.full((link.size(0),), -float('inf'), device=self._device)

        if valid_mask.any():
            dist_link = Categorical(logits=link_logits[valid_mask])
            lp1 = dist_link.log_prob(link[valid_mask])

            d_model = h_padded.size(-1)
            link_idx_exp = link[valid_mask].view(-1, 1, 1).expand(-1, 1, d_model)
            h_link = torch.gather(h_padded[valid_mask], dim=1, index=link_idx_exp).squeeze(1)

            rb_emb = self.rb_embedding(rb_idx[valid_mask])
            pb_input = torch.cat([h_link, rb_emb], dim=-1)
            pb_logits = self.pb_head(pb_input)
            pb_idx = power[valid_mask] * self._num_beam + beam[valid_mask]
            dist_pb = Categorical(logits=pb_logits)
            lp2 = dist_pb.log_prob(pb_idx)

            log_prob[valid_mask] = lp1 + lp2

        return log_prob

    def entropy(self, h_padded, link_logits, rb_idx):
        link_logits = torch.nan_to_num(link_logits, nan=-1e9)
        all_inf_mask = ~torch.isfinite(link_logits).any(dim=1)
        entropy = torch.zeros(link_logits.size(0), device=self._device)

        valid_idx = (~all_inf_mask).nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() > 0:
            dist_link = Categorical(logits=link_logits[valid_idx])
            link_idx = dist_link.sample()
            d_model = h_padded.size(-1)
            link_idx_exp = link_idx.view(-1, 1, 1).expand(-1, 1, d_model)
            h_link = torch.gather(h_padded[valid_idx], dim=1, index=link_idx_exp).squeeze(1)

            rb_emb = self.rb_embedding(rb_idx[valid_idx])
            pb_input = torch.cat([h_link, rb_emb], dim=-1)
            pb_logits = self.pb_head(pb_input)
            dist_pb = Categorical(logits=pb_logits)

            entropy[valid_idx] = dist_link.entropy() + dist_pb.entropy()

        return entropy
    
class ActDist:
    def __init__(self, logit, ptr, rb_idx, device):
        self._device = device
        self._ptr = ptr
        self._rb_idx = rb_idx
        self._batch_size = int(ptr.shape[0]) - 1
        self._num_power_level = logit.size(1)
        self._num_beam = logit.size(2)
        self._logit = logit

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
        for i, dist in enumerate(self._dist_list):
            if dist is not None:
                idx = int(dist.sample())
                node = idx // (self._num_power_level *self._num_beam)
                pow_beam = idx % (self._num_power_level *self._num_beam )
                power = pow_beam // (self._num_beam)
                beam = pow_beam % (self._num_beam)

                rb = int(self._rb_idx[self._ptr[i] + node].item())
            else:
                node, rb, power, beam = -1, -1, -1, -1
            action.append([node, rb, power, beam])
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
                node, rb, power, beam = a
                idx = node * self._num_beam * self._num_power_level +  power * self._num_beam + beam
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