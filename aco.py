import numpy as np
import random
import time
from ofdm_simulator.ofdm_simulator import OFDMSimulator
import matplotlib.pyplot as plt
import torch
import cupy as cp
import pandas as pd

class AntColonyOptimizer:
    def __init__(self, data_dir, tx_power_range, max_bs_power, noise_spectral_density, alpha,
                 num_ants, max_iterations, pheromone_importance, heuristic_importance, evaporation_rate, pheromone_constant, gpu=False):
        self._num_ants = num_ants
        self._max_iterations = max_iterations
        self._alpha = pheromone_importance
        self._beta = heuristic_importance
        self._rho = evaporation_rate
        self._Q = pheromone_constant
        self._sim = OFDMSimulator(data_dir=data_dir, tx_power_range=tx_power_range, max_bs_power=max_bs_power,
                                  noise_spectral_density=noise_spectral_density, alpha=alpha, num_ue_range=None, gpu=gpu)
        self.num_rb, self.num_beam, self.num_power_level = self._sim.num_rb, self._sim.num_beam, self._sim.num_power_level
        self.eval_file_list = self._sim.get_evaluation_networks_file_list()
        self.num_evaluation_networks = len(self.eval_file_list)

    def solve_all_evaluation_networks(self):
        perf_list = []
        elapsed_time_list = []
        for idx in range(self.num_evaluation_networks):
            perf, elapsed_time, solution = self.solve_evaluation_network(idx)
            perf_list.append(perf)
            elapsed_time_list.append(elapsed_time)
        avg_perf = sum(perf_list) / len(perf_list)
        avg_elapsed_time = sum(elapsed_time_list) / len(elapsed_time_list)
        return avg_perf, avg_elapsed_time

    def solve_evaluation_network(self, index):
        network = self._sim.get_evaluation_network(self.eval_file_list[index])
        perf, elapsed_time, solution = self.solve(network)
        return perf, elapsed_time, solution

    def solve(self, network):
        log_path = "aco_log.csv"
        start = time.perf_counter()
        ch = network['ch']
        num_link = ch.shape[0]
        num_entity = num_link * self.num_rb
        num_action = self.num_power_level * self.num_beam
        pheromone = np.ones((num_entity, num_action), dtype=np.float32)
        # weights = self.get_power_level_weights(self.num_power_level, self.num_beam)
        # pheromone *= weights[None, :]

        best_solution, best_perf = None, 0
        iteration_log = []
        best_perf_log = []
        feasible_ratio_log = []
        log_data = []
        for it in range(self._max_iterations):
            ant_solutions = []
            feasible_count = 0
            for ant in range(self._num_ants):
                # if ant == 1:
                #     power_level = pd.read_csv("power_level_result.csv").values[:,1:]
                #     beam_index = pd.read_csv("beam_index_result.csv").values[:,1:]
                #     solution = {
                #         'power_level': power_level,
                #         'beam_index': beam_index
                #     }
                if ant < 0.1 * self._num_ants:

                    solution = {
                        'power_level': np.random.randint(0, 2, (num_link, self.num_rb), dtype=np.int32),
                        'beam_index': np.random.randint(0, self.num_beam, (num_link, self.num_rb), dtype=np.int32)
                        }
                else:
                    solution = self.construct_solution(pheromone, num_link, network)
                if self._sim.is_solution_feasible(network, solution):
                    perf = self._sim.get_optimization_target(network, solution)
                    feasible_count += 1
                else:
                    perf = 0
                ant_solutions.append((solution, perf))
                if perf > best_perf:
                    best_solution, best_perf = solution, perf
            
            pheromone *= (1 - self._rho)
            ### default
            for sol, perf in ant_solutions:
                if perf <= 0.0:
                    continue 
                pl, bi = sol['power_level'], sol['beam_index']
                for l in range(num_link):
                    for rb in range(self.num_rb):
                        m = l * self.num_rb + rb
                        p = pl[l, rb]
                        b = bi[l, rb]
                        g = p * self.num_beam + b
                        pheromone[m, g] += self._Q * perf  # perf 그대로 반영


            # rank-based
            # sorted_ants = sorted(ant_solutions, key=lambda x: x[1], reverse=True)
            # top_k = sorted_ants[:5]  # 상위 5개 개미만 강화

            # for rank, (sol, _) in enumerate(top_k):
            #     rank_weight = (len(top_k) - rank) / len(top_k)  # 예: 1.0, 0.8, 0.6, ...
            #     pl, bi = sol['power_level'], sol['beam_index']
            #     for l in range(num_link):
            #         for rb in range(self.num_rb):
            #             m = l * self.num_rb + rb
            #             p, b = pl[l, rb], bi[l, rb]
            #             g = p * self.num_beam + b
            #             pheromone[m, g] += self._Q * rank_weight

            ### regulaization
            # perf_values = np.array([p for _, p in ant_solutions])
            # if len(perf_values) > 0:
            #     min_perf = np.min(perf_values)
            #     max_perf = np.max(perf_values)
            #     range_perf = max(max_perf - min_perf, 1e-8)

            #     for sol, perf in ant_solutions:
            #         if perf <= 0.0:
            #             continue
            #         norm_perf = 0.5 + 0.5 * (perf - min_perf) / range_perf  # scale to [0.5, 1.0]
            #         pl, bi = sol['power_level'], sol['beam_index']
            #         for l in range(num_link):
            #             for rb in range(self.num_rb):
            #                 m = l * self.num_rb + rb
            #                 p = pl[l, rb]
            #                 b = bi[l, rb]
            #                 g = p * self.num_beam + b
            #                 pheromone[m, g] += self._Q * norm_perf
            feasible_ratio = feasible_count / self._num_ants
            elapsed_time = time.perf_counter() - start
            log_data.append({
                "iteration": it,
                "elapsed_time_sec": elapsed_time,
                "best_performance": best_perf,
                "feasible_ratio": feasible_ratio
                })
            iteration_log.append(it)
            best_perf_log.append(best_perf)
            feasible_ratio_log.append(feasible_ratio)
            print(f"Iteration {it}: Best performance = {best_perf:.4f} | Feasible ants: {feasible_count}/{self._num_ants} ({feasible_ratio:.2%})")
        df_log = pd.DataFrame(log_data)
        df_log.to_csv(log_path, index=False)
        end = time.perf_counter()
        total_elapsed_time = end - start
        self.plot_result(iteration_log, best_perf_log, feasible_ratio_log)
        return best_perf, total_elapsed_time, best_solution

    def construct_solution(self, pheromone, num_link, network):
        power_level = np.zeros((num_link, self.num_rb), dtype=np.int32)
        beam_index = np.zeros_like(power_level)

        # heuristic = np.zeros_like(pheromone)
        # ch = network['ch']  # (link, bs, rb, beam)
        # assoc = network['assoc']

        # for l in range(num_link):
        #     for rb in range(self.num_rb):
        #         m = l * self.num_rb + rb
        #         bs = assoc[l]
        #         for p in range(2):  # power_level ∈ {0, 1}
        #             if p == 0:
        #                 continue  # skip power off
        #             for b in range(self.num_beam):
        #                 g = p * self.num_beam + b
        #                 gain = ch[l, bs, rb, b]
        #                 heuristic[m, g] = gain + 1e-10  # ensure non-zero
        pheromone = pheromone ** self._alpha
        # heuristic = heuristic ** self._beta
        #prob_matrix = pheromone * heuristic
        prob_matrix = pheromone ** self._alpha
        prob_matrix /= np.sum(prob_matrix, axis=1, keepdims=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        g_indices = torch.multinomial(torch.tensor(prob_matrix, device=device), num_samples=1).squeeze(-1)

        power_level_flat = g_indices // self.num_beam
        beam_index_flat = g_indices % self.num_beam

        power_level= power_level_flat.reshape((num_link, self.num_rb))
        beam_index = beam_index_flat.reshape((num_link, self.num_rb))

        return {'power_level': power_level.cpu().numpy(), 'beam_index': beam_index.cpu().numpy()}

    def get_power_level_weights(self, num_power_level, num_beam, peak_weight=10.0, min_weight=0.5, decay_rate=2):
        x = np.arange(num_power_level)
        normalized = x / (num_power_level - 1)
        power_weights = peak_weight * np.exp(-decay_rate * normalized)
        power_weights = np.clip(power_weights, min_weight, peak_weight)

        # power level별 weight를 각 beam 방향 수만큼 반복해서 action 단위로 확장
        action_weights = np.repeat(power_weights, num_beam)  # shape: (num_power_level * num_beam,)
        return action_weights

    def plot_result(self, iter, perf, feasible_ratio):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(iter, perf, color='blue', label='Best Performance')
        plt.xlabel('Iteration')
        plt.ylabel('Performance')
        plt.title('ACO Performance over Iterations')
        plt.grid(True)
        plt.legend()

        # Feasible ratio plot
        plt.subplot(1, 2, 2)
        plt.plot(iter, feasible_ratio, color='green', label='Feasible Ratio')
        plt.xlabel('Iteration')
        plt.ylabel('Feasible Ant Ratio')
        plt.title('Feasible Solutions over Iterations')
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    data_dir = 'myeongdong_arr_4_rb_16'
    tx_power_range = {'max_power': 2, 'num_power_level': 4}
    max_bs_power = 10
    noise_spectral_density = -174.0
    alpha = 0.0
    gpu = True

    num_ants = 50
    max_iterations = 1000
    pheromone_importance = 1
    heuristic_importance = 0
    evaporation_rate = 0.01
    pheromone_constant = 10

    alg = AntColonyOptimizer(data_dir, tx_power_range, max_bs_power, noise_spectral_density, alpha,
                              num_ants, max_iterations, pheromone_importance, heuristic_importance, evaporation_rate, pheromone_constant, gpu)
    perf, elapsed_time, solution = alg.solve_evaluation_network(0)
    print(f'Performance: {perf}, Elapsed time: {elapsed_time}')