from argparse import ArgumentParser
import os, signal
from utils import get_problem_names


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--problem", type=str, default=["bstdiag"], nargs="+", help="problems to test"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=["raqneirs", "qnehvi"],
        nargs="+",
        help="algorithms to test",
    )
    parser.add_argument(
        "--n-seed", type=int, default=1, help="number of different seeds"
    )
    parser.add_argument(
        "--subfolder", type=str, default="default", help="subfolder of result"
    )
    parser.add_argument(
        "--savefig",
        default=False,
        action="store_true",
        help="saving as png instead of showing the plot",
    )
    args = parser.parse_args()

    args_str = f" --n-seed {args.n_seed} --subfolder {args.subfolder}"
    if args.problem is not None:
        args_str += " --problem " + " ".join(args.problem)
    if args.algo is not None:
        args_str += " --algo " + " ".join(args.algo)
    if args.savefig:
        args_str += " --savefig"

    command = "python ./visualization/visualize_performance_space_batch.py" + args_str
    ret_code = os.system(command)
    command = "python ./visualization/visualize_function_space_batch.py" + args_str
    ret_code = os.system(command)
    command = "python ./visualization/visualize_hv_batch.py" + args_str
    ret_code = os.system(command)
    command = "python ./visualization/visualize_pf.py" + args_str
    ret_code = os.system(command)


if __name__ == "__main__":
    main()
