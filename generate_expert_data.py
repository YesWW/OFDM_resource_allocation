import torch
import dill
from tqdm import tqdm
from tabu_search import TabuSearch

def generate_and_save_expert_data(save_path, num_networks=100):
    # Tabu Search 초기화
    data_dir = 'myeongdong_arr_4_rb_16'  # 설정에 맞게
    tx_power_range = {'max_power': 2, 'num_power_level': 16}
    max_bs_power = 10
    noise_spectral_density = -174.0
    alpha = 0.0
    tabu_move_selection_prob = 0.01
    tabu_max_iterations = 300
    tabu_tenure = 100
    gpu = True

    tabu = TabuSearch(data_dir, tx_power_range, max_bs_power, noise_spectral_density, alpha,
                      tabu_move_selection_prob, tabu_max_iterations, tabu_tenure, gpu)
    tabu._sim.num_ue_range = [60, 80]
    tabu._sim.num_rb = 16

    networks = tabu._sim.get_networks(num_networks)
    expert_data = []

    for net in tqdm(networks):
        _, _, solution = tabu.solve(net)
        expert_data.append({'network': net, 'solution': solution})

    torch.save(expert_data, save_path)
    print(f"Saved {len(expert_data)} expert solutions to {save_path}")

if __name__ == '__main__':
    save_path = './expert_data.pt'
    generate_and_save_expert_data(save_path, num_networks=200)