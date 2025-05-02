import os
import dill
import torch
from torch.utils.data import Dataset

directory_path = 'expert_dataset'
power_list = []
beam_list = []
action_list = []

for file in os.listdir(directory_path):
    if file.endswith(".pkl") and file.startswith("tabu_trajectory_"):
        with open(os.path.join(directory_path, file), 'rb') as f:
            traj_list = dill.load(f)
            #t = torch.tensor(traj_list)
            print(1)


print(1)