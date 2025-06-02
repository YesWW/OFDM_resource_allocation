import numpy as np
import random
import time
from ofdm_simulator.ofdm_simulator_rb import OFDMSimulator
import pandas as pd

class TabuSearch:
    def __init__(self, data_dir, tx_power_range, max_bs_power, noise_spectral_density, alpha,
                 tabu_move_selection_prob, tabu_max_iterations, tabu_tenure, gpu=False):
        self._tabu_move_selection_prob = tabu_move_selection_prob
        self._tabu_max_iterations = tabu_max_iterations
        self._tabu_tenure = tabu_tenure
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
        log_path = "tabu_log.csv"
        start = time.perf_counter()
        ch = network['ch']
        num_link = ch.shape[0]
        tabu_list = {}
        all_moves = self.get_all_moves(num_link)
        cur_solution = self.get_initial_solution(num_link)
        best_solution, best_perf = None, -np.inf
        log_data = []
        for it in range(self._tabu_max_iterations):
            cand_moves, cand_solutions = self.apply_move(cur_solution, all_moves)
            selected_solution, selected_move, selected_perf = None, None, -np.inf
            for cand_move, cand_solution in zip(cand_moves, cand_solutions):
                feasible = self._sim.is_solution_feasible(network, cand_solution)
                if feasible:
                    cand_perf = self._sim.get_optimization_target(network, cand_solution)
                    if cand_perf > selected_perf and ((cand_move not in tabu_list) or cand_perf > best_perf):
                        selected_solution, selected_move, selected_perf = cand_solution, cand_move, cand_perf
                    if cand_perf > best_perf:
                        best_solution, best_perf = cand_solution, cand_perf
            if selected_solution is None:
                break
            cur_solution = selected_solution
            tabu_list[selected_move] = self._tabu_tenure
            tabu_list = {move:(tenure - 1) for move, tenure in tabu_list.items() if tenure > 1}
            print(f"iteration: {it}, performance: {best_perf}")
            elapsed_time = time.perf_counter() - start
            log_data.append({
                "iteration": it,
                "elapsed_time_sec": elapsed_time,
                "best_performance": best_perf,
                })
        df_p = pd.DataFrame(cur_solution['power_level'])
        df_b = pd.DataFrame(cur_solution['beam_index'])
        df_p.to_csv("power_level_result.csv")
        df_b.to_csv("beam_index_result.csv")
        end = time.perf_counter()
        df_log = pd.DataFrame(log_data)
        df_log.to_csv(log_path, index=False)
        total_elapsed_time = end - start
        return best_perf, total_elapsed_time, best_solution

    def get_all_moves(self, num_link):
        link, rb, pwr, beam = (np.arange(num_link), np.arange(self.num_rb),
                               np.arange(self.num_power_level), np.arange(self.num_beam))
        link, rb, pwr, beam = np.meshgrid(link, rb, pwr, beam, indexing='ij')
        moves = np.stack([link, rb, pwr, beam], axis=-1).reshape(-1, 4)
        moves = [tuple(move) for move in moves]
        return moves

    def get_initial_solution(self, num_link):
        power_level = np.zeros((num_link, self.num_rb)).astype(np.int32)
        beam_index = np.zeros((num_link, self.num_rb)).astype(np.int32)
        return {'power_level': power_level, 'beam_index': beam_index}

    def apply_move(self, cur_solution, all_moves):
        cand_moves = []
        cand_solutions = []
        n_moves = int(len(all_moves) * self._tabu_move_selection_prob)
        selected_moves = random.sample(all_moves, n_moves)
        for move in selected_moves:
            pl, bi = cur_solution['power_level'].copy(), cur_solution['beam_index'].copy()
            pl[move[0], move[1]] = move[2]
            bi[move[0], move[1]] = move[3]
            cand_moves.append(move)
            cand_solutions.append({'power_level': pl, 'beam_index': bi})
        return cand_moves, cand_solutions


if __name__ == '__main__':
    data_dir = 'myeongdong_arr_4_rb_16'
    tx_power_range = {'max_power': 2, 'num_power_level': 4}  # Power level quantization
    max_bs_power = 10  # Maximum base station tx power (watt)
    noise_spectral_density = -174.0  # Noise spectral density (dBm/Hz)
    alpha = 0.0  # coefficient for alpha-fairness function (0.0: sum, 1.0: proportional, inf: max-min)
    gpu = True
    tabu_move_selection_prob = 0.01
    tabu_max_iterations = 1000
    tabu_tenure = 100
    alg = TabuSearch(data_dir, tx_power_range, max_bs_power, noise_spectral_density, alpha,
                     tabu_move_selection_prob, tabu_max_iterations, tabu_tenure, gpu)
    perf, elapsed_time, solution = alg.solve_evaluation_network(0)
    print(f'Performance: {perf}, Elapsed time: {elapsed_time}')


