from argparse import ArgumentParser
import os, signal
from utils import get_problem_names


def main():
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, nargs='+', default=None, help='problems to test')
    parser.add_argument('--algo', type=str, nargs='+', default=None, help='algorithms to test')
    parser.add_argument('--n-seeds', type=int, default=8, help='number of different seeds')
    parser.add_argument('--subfolder', type=str, default='default', help='subfolder of result')
    parser.add_argument('--savefig', default=False, action='store_true', help='saving as png instead of showing the plot')
    args = parser.parse_args()

    if args.problem is None: args.problem = get_problem_names()

    for problem in args.problem:
        for seed in range(args.n_seeds):
            command = f'python visualization/visualize_function_space.py \
                --problem {problem} \
                --seed {seed} --subfolder {args.subfolder}'

            if args.algo is not None: command += ' --algo ' + ' '.join(args.algo) 
            if args.savefig: command += ' --savefig'

            ret_code = os.system(command)
        

if __name__ == "__main__":
    main()
