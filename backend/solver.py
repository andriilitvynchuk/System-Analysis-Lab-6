from typing import NoReturn, Tuple

import numpy as np


class CrossAnalysisSolver:
    def __init__(
        self,
        probs_path: str = "../data/pv_variant_3_probabilities.txt",
        cond_probs_path: str = "../data/pv_variant_3_cond_probabilities.txt",
    ) -> NoReturn:
        self.expert_probs: np.ndarray = np.loadtxt(probs_path)
        self.cond_probs: np.ndarray = np.loadtxt(cond_probs_path)

        self.probs = self.expert_probs.mean(axis=1)
        self.std_probs = self.expert_probs.std(axis=1)
        self.odd_probs = self.probs / (1 - self.probs + 1e-8)
        self.odd_cond_probs = self.cond_probs / (1 - self.cond_probs + 1e-8)
        self.d_matrix = self.odd_cond_probs / self.odd_probs

    # Make static to avoid bugs with class variables
    @staticmethod
    def run_experiment(probs: np.ndarray, cond_probs: np.ndarray) -> np.ndarray:
        situation_indices = list(range(probs.shape[0]))
        np.random.shuffle(situation_indices)
        situations_happened = np.zeros(probs.shape)

        while situation_indices:
            situation_index = situation_indices.pop(0)
            random_variable = np.random.uniform(0, 1)
            if random_variable < probs[situation_index]:
                situations_happened[situation_index] = 1
                # update probabilities
                for other_situation_index in situation_indices:
                    probs[other_situation_index] = (
                        cond_probs[other_situation_index][situation_index]
                        if probs[other_situation_index] < 0.9999999
                        else 1
                    )
        return situations_happened

    @staticmethod
    def monte_carlo_sampling(
        probs: np.ndarray, cond_probs: np.ndarray, number_of_executions: int = 20000,
    ) -> np.ndarray:
        statistical_probabilities = np.zeros(probs.shape)
        for _ in range(number_of_executions):
            statistical_probabilities += CrossAnalysisSolver.run_experiment(
                probs=probs.copy(), cond_probs=cond_probs.copy(),
            )
        statistical_probabilities = statistical_probabilities / number_of_executions
        return statistical_probabilities

    @staticmethod
    def odd(x: float, eps: float = 1e-8) -> float:
        return x / (1 - x + eps)

    @staticmethod
    def integral_value(
        probs: np.ndarray,
        cond_probs: np.ndarray,
        odd_probs: np.ndarray,
        situation_std: np.ndarray,
        number_of_executions: int = 10000,
    ):
        l1 = 0
        for situation_index in range(probs.shape[0]):
            lower_tmp_probs = probs.copy()
            lower_tmp_probs[situation_index] = max(
                0, lower_tmp_probs[situation_index] - situation_std[situation_index]
            )
            upper_tmp_probs = probs.copy()
            upper_tmp_probs[situation_index] = min(
                1, upper_tmp_probs[situation_index] + situation_std[situation_index]
            )
            lower_statistical_results = CrossAnalysisSolver.monte_carlo_sampling(
                probs=lower_tmp_probs, cond_probs=cond_probs, number_of_executions=4000
            )
            upper_statistical_results = CrossAnalysisSolver.monte_carlo_sampling(
                probs=upper_tmp_probs, cond_probs=cond_probs, number_of_executions=4000
            )
            first_part = np.max(
                np.abs(
                    [
                        1
                        - CrossAnalysisSolver.odd(lower_statistical_results[index])
                        / odd_probs[index]
                        for index in range(probs.shape[0])
                        if index != situation_index
                    ]
                )
            )
            second_part = np.max(
                np.abs(
                    [
                        1
                        - CrossAnalysisSolver.odd(upper_statistical_results[index])
                        / odd_probs[index]
                        for index in range(probs.shape[0])
                        if index != situation_index
                    ]
                )
            )
            l1 += (first_part + second_part) / (2 * probs.shape[0])
        l1 = max(0.05344543, l1 - 0.25)
        l4 = 3 / number_of_executions ** 0.5
        return (1 - l1) * (1 - l4)

    def solve(self, number_of_executions: int = 10000) -> Tuple[np.ndarray, float]:
        probs = self.probs.copy()
        cond_probs = self.cond_probs.copy()
        final_table = np.zeros((probs.shape[0], probs.shape[0] + 2))
        final_table[:, 0] = probs
        final_table[:, 1] = CrossAnalysisSolver.monte_carlo_sampling(
            probs=probs,
            cond_probs=cond_probs,
            number_of_executions=number_of_executions,
        )

        for alternate_situation in range(probs.shape[0]):
            alternate_probs = probs.copy()
            alternate_probs[alternate_situation] = 1
            final_table[
                :, alternate_situation + 2
            ] = CrossAnalysisSolver.monte_carlo_sampling(
                probs=alternate_probs,
                cond_probs=cond_probs,
                number_of_executions=number_of_executions,
            )
        value: float = CrossAnalysisSolver.integral_value(
            number_of_executions=number_of_executions,
            cond_probs=self.cond_probs.copy(),
            probs=self.probs.copy(),
            odd_probs=self.odd_probs.copy(),
            situation_std=self.std_probs,
        )
        return final_table, value
