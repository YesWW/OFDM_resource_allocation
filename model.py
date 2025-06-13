import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.init import xavier_uniform_, constant_
from torch.distributions.categorical import Categorical
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import global_mean_pool

class Policy(nn.Module):
    def __init__(self, num_power_level, num_rb, num_beam, power_attn_num_level, model_params, device):
        super(Policy, self).__init__()
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
            dim_feedforward=self._dim_feedforward, dropout=self._dropout,
            activation="relu", device=self._device
        )
        self._logit_linear = Linear(self._d_model, self._num_power_level * self._num_beam, device=device)

        self._link_embedding = nn.Embedding(num_embeddings=num_rb, embedding_dim=self._d_model, device=device)
        self._rb_embedding = PositionalEncoding(d_model=self._d_model, max_len=num_rb, device=device)

        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self._logit_linear.weight)
        nn.init.constant_(self._logit_linear.bias, 0.)

    def forward(self, power_alloc, beam_alloc, node_power_attn, edge_power_attn, edge_index, ptr, batch, logit_mask=None):
        resource_alloc = torch.cat([power_alloc, beam_alloc], dim=2).reshape(power_alloc.size(0), -1)
        node_power_attn = node_power_attn.reshape(node_power_attn.size(0), -1)
        edge_power_attn = edge_power_attn.reshape(edge_power_attn.size(0), -1)
        x = self._graph_transformer(input=resource_alloc, node_embedding=node_power_attn,
                                    edge_attr=edge_power_attn, edge_index=edge_index)

        
        logit = self._logit_linear(x)

        if logit_mask is not None:
            logit_mask = torch.gather(logit_mask, dim=1, index=1).squeeze(1).unsqueeze(-1)
            logit = torch.where(logit_mask, logit, other=-torch.inf)

        return logit

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device='cpu'):
        super().__init__()
        # [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        # [d_model//2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
            (-torch.log(torch.tensor(10000.0, device=device)) / d_model)
        )

        pe = torch.zeros(max_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # even index
        pe[:, 1::2] = torch.cos(position * div_term)  # odd index

        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, index):
        """
        index: 정수 scalar or (B,) shape의 텐서
        return: positional vector(s) → shape: (B, d_model) or (d_model,)
        """
        return self.pe[index]


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