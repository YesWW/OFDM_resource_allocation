import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import numpy as np

class Buffer:
    def __init__(self, graph, power_alloc, beam_alloc, action, act_log_prob, value, ongoing, target, link_rb, device):
        self._device = device
        self._ptr = graph['ptr']
        self._graph_list = graph.to_data_list()
        self._batch_size = len(self._graph_list)
        self._power_alloc = power_alloc
        self._beam_alloc = beam_alloc
        self._action = action
        self._act_log_prob = act_log_prob
        self._value = value
        self._ongoing = ongoing
        self._target = target
        self._link_rb = link_rb

        self._reward = None
        self._return = None
        self._idx = []
        num_step = torch.sum(self._ongoing.int(), dim=0)
        for s in range(self._batch_size):
            self._idx.append(np.stack((s * np.ones((num_step[s],)), np.arange(0, num_step[s])), axis=1).astype(int))
        self._idx = np.concatenate(self._idx, axis=0)

    def __len__(self):
        return self._idx.shape[0]

    def __getitem__(self, idx):
        samp, step = self._idx[idx][0], self._idx[idx][1]
        out = {}
        out['graph'] = self._graph_list[samp].to(self._device)
        out['power_alloc'] = self._power_alloc[step, self._ptr[samp]: self._ptr[samp+1], :, :].to(self._device)
        out['beam_alloc'] = self._beam_alloc[step, self._ptr[samp]: self._ptr[samp+1], :, :].to(self._device)
        out['action'] = self._action[step, samp].to(self._device)
        out['act_log_prob'] = self._act_log_prob[step, samp].to(self._device)
        out['value'] = self._value[step, samp].to(self._device)
        out['ongoing'] = self._ongoing[step, samp].to(self._device)
        out['link_rb'] = self._link_rb[step, samp].to(self._device)
        if self._reward is not None:
            out['reward'] = self._reward[step, samp].to(self._device)
        if self._return is not None:
            out['return'] = self._return[step, samp].to(self._device)
        return out

    def cal_reward(self):
        self._reward = torch.zeros_like(self._target)
        self._reward[1:] = self._target[1:] - self._target[:-1]
        self._reward[0] = torch.zeros_like(self._reward[0])
        #self._reward *= 100
        #self._reward = self._target.clone()

    def get_performance(self):
        return self._reward

    def cal_lambda_return(self, gamma, lamb):
        num_step = self._reward.size(0)-1
        ret = torch.zeros((num_step, self._batch_size))
        ret[num_step - 1, :] = self._reward[num_step, :]
        value = torch.where(self._ongoing, self._value, 0.0)
        for step in range(num_step - 2, -1, -1):
            ret[step, :] = ((1 - lamb) * gamma * value[step + 1, :] + self._reward[step + 1, :]
                            + lamb * gamma * ret[step + 1, :])
        dummy_row = torch.zeros((1, self._batch_size), device=ret.device)
        self._return = torch.cat([dummy_row, ret], dim=0)



def collate_fn(samp):
    keys = list(samp[0].keys())
    out = {k: [] for k in keys}
    for s in samp:
        for k in keys:
            out[k].append(s[k])
    out['graph'] = Batch.from_data_list(out['graph'])
    out['power_alloc'] = torch.concatenate(out['power_alloc'], dim=0)
    out['beam_alloc'] = torch.concatenate(out['beam_alloc'], dim=0)
    out['action'] = torch.stack(out['action'], dim=0)
    out['act_log_prob'] = torch.stack(out['act_log_prob'], dim=0)
    out['value'] = torch.stack(out['value'], dim=0)
    out['ongoing'] = torch.stack(out['ongoing'], dim=0)
    out['link_rb'] = torch.stack(out['link_rb'], dim=0)
    if 'reward' in keys:
        out['reward'] = torch.stack(out['reward'], dim=0)
    if 'return' in keys:
        out['return'] = torch.stack(out['return'], dim=0)
    return out


def get_buffer_dataloader(buf, batch_size, shuffle=True):
    dataloader = DataLoader(buf, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    pass