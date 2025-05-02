import numpy as np
import time
import random
import pandas as pd
from pathlib import Path
from ofdm_simulator.ofdm_simulator_4 import OFDMSimulator

class GreedySolverOFDM:
    def __init__(self, data_dir, tx_power_range, max_bs_power,
                 noise_spectral_density, alpha, num_ue_range,
                 gpu=False, device="cpu"):
        
        self._sim = OFDMSimulator(data_dir=data_dir,
                                  tx_power_range=tx_power_range,
                                  max_bs_power=max_bs_power,
                                  noise_spectral_density=noise_spectral_density,
                                  alpha=alpha,
                                  num_ue_range=num_ue_range,
                                  gpu=gpu)
        
        self._num_rb = self._sim.num_rb
        self._num_beam = self._sim.num_beam
        self._num_power_level = self._sim.num_power_level
        self._device = device

        self.eval_file_list = self._sim.get_evaluation_networks_file_list()
        self.num_evaluation_networks = len(self.eval_file_list)

    def solve_all_evaluation_networks(self):
        perf_list, elapsed_time_list = [], []
        for idx in range(self.num_evaluation_networks):
            perf, elapsed_time, _ = self.solve_evaluation_network(idx)
            perf_list.append(perf)
            elapsed_time_list.append(elapsed_time)
        return np.mean(perf_list), np.sum(elapsed_time_list)

    def solve_evaluation_network(self, index):
        network = self._sim.get_evaluation_network(Path(self.eval_file_list[index]).name)
        return self.solve(network)

    def solve(self, network):
        start = time.perf_counter()
        num_ue = network['ch'].shape[0]

        power_level = np.zeros((num_ue, self._num_rb), dtype=int)
        beam_index = np.random.randint(0, self._num_beam, size=(num_ue, self._num_rb))
        solution = {'power_level': power_level, 'beam_index': beam_index}

        best_perf = self._sim.get_optimization_target(network, solution)
        best_solution = solution.copy()
        output = []

        for it in range(num_ue):
            candidate_solutions = self.get_all_moves(solution, num_ue)
            for cand in candidate_solutions:
                if not self._sim.is_solution_feasible(network, cand):
                    continue
                perf = self._sim.get_optimization_target(network, cand)
                if perf > best_perf:
                    best_perf = perf
                    best_solution = cand
            solution = best_solution
            iter_time = time.perf_counter() - start
            output.append([it, best_perf, iter_time])
            print(f"Iteration {it} | Num ue :{num_ue} | Target: {best_perf:.4f}, Time: {iter_time:.2f}s")

        elapsed = time.perf_counter() - start
        result_path = Path(__file__).parents[0] / 'evaluation_results' / 'result_greedy_ofdm.csv'
        result_path.parent.mkdir(exist_ok=True)
        pd.DataFrame(output, columns=['iter', 'target', 'time']).to_csv(result_path, index=False)
        return best_perf, elapsed, best_solution

    def get_all_moves(self, solution, num_ue):
        candidates = []
        for ue in range(num_ue):
            for rb in range(self._num_rb):
                cur_p = solution['power_level'][ue, rb]
                if cur_p < self._num_power_level - 1:
                    new_p = cur_p + 1
                    for b in range(self._num_beam):
                        new_sol = {
                            'power_level': solution['power_level'].copy(),
                            'beam_index': solution['beam_index'].copy()
                        }
                        new_sol['power_level'][ue, rb] = new_p
                        new_sol['beam_index'][ue, rb] = b
                        candidates.append(new_sol)
        return candidates


if __name__ == '__main__':
    data_dir = 'myeongdong_arr_4_rb_16'
    tx_power_range = {'max_power': 2, 'num_power_level': 4}
    max_bs_power = 10
    noise_spectral_density = -174.0
    alpha = 0.0
    num_ue_range = [20, 40]
    gpu = True

    alg = GreedySolverOFDM(data_dir=data_dir,
                           tx_power_range=tx_power_range,
                           max_bs_power=max_bs_power,
                           noise_spectral_density=noise_spectral_density,
                           alpha=alpha,
                           num_ue_range=num_ue_range,
                           gpu=gpu,
                           device="cuda:0")

    perf, elapsed_time, _ = alg.solve_evaluation_network(0)
    print(f'Final Performance: {perf:.4f}, Time: {elapsed_time:.2f}s')
