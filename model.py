import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.init import xavier_uniform_, constant_
from torch.distributions.categorical import Categorical
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import global_mean_pool

class AC(nn.Module):
    def __init__(self, num_power, num_rb, num_beam, power_attn_num_level, model_params, device):
        super(AC, self).__init__()
        self._device = device
        self._num_power = num_power
        self._num_rb = num_rb
        self._num_beam = num_beam
        self._power_attn_num_level = power_attn_num_level
        self._d_model = model_params['d_model']
        self._n_head = model_params['n_head']
        self._dim_feedforward = model_params['dim_feedforward']
        self._num_layers = model_params['actor_num_layers']
        self._dropout = model_params['dropout']

        self._graph_transformer = GraphTransformer(
            input_dim=(self._num_power + self._num_beam) * self._num_rb,
            embedding_dim=self._power_attn_num_level * self._num_rb * self._num_beam,
            num_layers=self._num_layers, d_model=self._d_model, n_head=self._n_head,
            edge_dim=self._power_attn_num_level * self._num_rb * self._num_beam,
            dim_feedforward=self._dim_feedforward, dropout=self._dropout,
            activation="relu", device=self._device
        )

        self._rb_emb = nn.Embedding(num_embeddings=num_rb, embedding_dim=self._d_model).to(device)
        self._actor_linear = Linear(3*self._d_model, num_power * num_beam, device=device)
        self._critic_linear = Linear(self._d_model, 1, device=device)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self._actor_linear.weight)
        nn.init.constant_(self._actor_linear.bias, 0.)
        nn.init.xavier_uniform_(self._critic_linear.weight)
        nn.init.constant_(self._critic_linear.bias, 0.)

    def forward(self, power_alloc, beam_alloc, node_power_attn, edge_power_attn, edge_index, ptr, batch, link_rb):
        resource_alloc = torch.cat([power_alloc, beam_alloc], dim=2).reshape(power_alloc.size(0), -1)
        node_power_attn = node_power_attn.reshape(node_power_attn.size(0), -1)
        edge_power_attn = edge_power_attn.reshape(edge_power_attn.size(0), -1)

        x = self._graph_transformer(input=resource_alloc, node_embedding=node_power_attn,
                                    edge_attr=edge_power_attn, edge_index=edge_index)
        
        global_mean = global_mean_pool(x=x, batch=batch)
        value = self._critic_linear(global_mean)[:, 0]
        link_emb = x[link_rb[:,0]]
        rb_emb = self._rb_emb(link_rb[:,1])
        x = torch.cat([global_mean, link_emb, rb_emb], dim=1)
        logit = self._actor_linear(x)   # [batch, power*beam]
        dist = Categorical(logits=logit)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_probs, entropy, value

class ActDist:
    def __init__(self, logit, ptr, device):
        self._logit = logit
        self._device = device
        self._ptr = ptr
        self._batch_size = int(ptr.shape[0]) - 1
        self._num_rb = logit.size(1)
        self._num_power = logit.size(2)
        self._num_beam = logit.size(3)
        
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
                node = idx // (self._num_rb * self._num_power * self._num_beam)
                rbpb = idx % (self._num_rb * self._num_power * self._num_beam)
                rb = rbpb // (self._num_power * self._num_beam)
                pb = rbpb % (self._num_power * self._num_beam)
                power = pb // self._num_power
                beam = pb % self._num_beam
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
                idx = node * self._num_rb * self._num_power * self._num_beam + rb * self._num_power * self._num_beam + power * self._num_beam + beam
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