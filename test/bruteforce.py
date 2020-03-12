import sys
from typing import NoReturn


sys.path.append("..")
if True:
    from backend import CrossAnalysisSolver


def main() -> NoReturn:
    solver = CrossAnalysisSolver()
    solver.solve()


if __name__ == "__main__":
    main()
