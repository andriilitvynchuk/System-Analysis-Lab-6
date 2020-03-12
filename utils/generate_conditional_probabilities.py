import argparse
from typing import NoReturn

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prob-file", "-p", default="../data/pv_variant_3_probabilities.txt", type=str
    )
    parser.add_argument(
        "--output-file",
        "-o",
        default="../data/pv_variant_3_cond_probabilities.txt",
        type=str,
    )
    parser.add_argument("--seed", "-s", default=8, type=int)
    args: argparse.Namespace = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> NoReturn:
    np.random.seed(args.seed)
    expert_probs: np.ndarray = np.loadtxt(args.prob_file)
    probs = expert_probs.mean(axis=1)
    cond_probs: np.ndarray = np.eye(probs.shape[0])
    for i in range(cond_probs.shape[0]):
        for j in range(cond_probs.shape[1]):
            if i != j:
                upper_boundary: float = min(probs[i] / probs[j], 1)
                lower_boundary: float = max((probs[i] - 1 + probs[j]) / probs[i], 0)
                if upper_boundary < 0 or lower_boundary > 1:
                    raise ValueError(
                        f"Something wrong with this range: [{lower_boundary} , {upper_boundary}]"
                    )
                cond_probs[i][j] = np.random.uniform(lower_boundary, upper_boundary)
    np.savetxt(args.output_file, cond_probs, fmt="%.4f")


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
