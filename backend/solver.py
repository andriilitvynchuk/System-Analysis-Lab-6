from typing import Any, Dict, List, NoReturn, Optional

import numpy as np


class CrossAnalysisSolver:
    def __init__(
        self,
        probs_path: str = "../data/pv_variant_3_probabilities.txt",
        cond_probs_path: str = "../data/pv_variant_3_cond_probabilities.txt",
        number_of_executions: int = 20000,
    ) -> NoReturn:
        self.expert_probs: np.ndarray = np.loadtxt(probs_path)
        self.cond_probs: np.ndarray = np.loadtxt(cond_probs_path)
        self.number_of_executions = number_of_executions

        self.probs = self.expert_probs.mean(axis=1)
        self.odd_probs = self.probs / (1 - self.probs + 1e-8)
        self.odd_cond_probs = self.cond_probs / (1 - self.cond_probs + 1e-8)
        self.d_matrix = self.odd_cond_probs / self.odd_probs
        # self.d_matrix = self.odd_cond_probs / self.odd_probs.reshape(1, -1)

    # Make static to avoid bugs with class variables
    @staticmethod
    def run_experiment(
        probs: np.ndarray,
        cond_probs: np.ndarray,
        odd_probs: np.ndarray,
        odd_cond_probs: np.ndarray,
        d_matrix: np.ndarray,
    ) -> np.ndarray:
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
                    probs[other_situation_index] = cond_probs[other_situation_index][
                        situation_index
                    ]
                    # odd_probs[other_situation_index] = (
                    #     odd_probs[other_situation_index]
                    #     * d_matrix[other_situation_index][situation_index]
                    # )
                    # probs[other_situation_index] = odd_probs[other_situation_index] / (
                    #     1 + odd_probs[other_situation_index]
                    # )
        return situations_happened

    def monte_carlo_sampling(self, begin_probs: Optional[np.ndarray] = None):
        probs = begin_probs if begin_probs is not None else self.probs
        statistical_probabilities = np.zeros(probs.shape)
        for _ in range(self.number_of_executions):
            statistical_probabilities += CrossAnalysisSolver.run_experiment(
                probs=probs.copy(),
                cond_probs=self.cond_probs.copy(),
                odd_probs=self.odd_probs.copy(),
                odd_cond_probs=self.odd_cond_probs.copy(),
                d_matrix=self.d_matrix.copy(),
            )
        print(probs, statistical_probabilities / self.number_of_executions)
