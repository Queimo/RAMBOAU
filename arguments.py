import os
from argparse import ArgumentParser, Namespace
import yaml
from multiprocessing import cpu_count

'''
Get argument values from command line
Here we speficy different argument parsers to avoid argument conflict between initializing each components
'''

alpha = 0.9
n_init_sample = 6
n_iter = 20
batch_size = 1
n_w = 11
 

def get_general_args(args=None):
    '''
    General arguments: problem and algorithm description, experiment settings
    '''
    parser = ArgumentParser()

    parser.add_argument('--problem', type=str, default='bstdiag', 
        help='optimization problem')
    parser.add_argument('--n-var', type=int, default=2, 
        help='number of design variables')
    parser.add_argument('--n-obj', type=int, default=2, 
        help='number of objectives')
    parser.add_argument('--n-init-sample', type=int, default=n_init_sample, 
        help='number of initial design samples')
    parser.add_argument('--n-iter', type=int, default=n_iter, 
        help='number of optimization iterations')
    parser.add_argument('--ref-point', type=float, nargs='+', default=None, 
        help='reference point for calculating hypervolume')
    parser.add_argument('--batch-size', type=int, default=batch_size, 
        help='size of the selected batch in one iteration')
    parser.add_argument('--seed', type=int, default=0, 
        help='random seed')
    parser.add_argument('--n-seed', type=int, default=1,
        help='number of random seeds / test runs')
    parser.add_argument('--auto-ref-point', type=bool, default=True,
        help='automatically calculate reference point based on model prediction')

    parser.add_argument('--algo', type=str, default='raqneirs',
        help='type of algorithm to use with some predefined arguments, or custom arguments')

    parser.add_argument('--subfolder', type=str, default='default',
        help='subfolder name for storing results, directly store under result/ as default')
    parser.add_argument('--exp-name', type=str, default=None,
        help='custom experiment name to distinguish between experiments on same problem and same algorithm')
    parser.add_argument('--log-to-file', default=False, action='store_true',
        help='log output to file rather than print by stdout')
    parser.add_argument('--n-process', type=int, default=cpu_count(),
        help='number of processes to be used for parallelization')

    args, _ = parser.parse_known_args(args)
    return args


def get_surroagte_args(args=None):
    '''
    Arguments for fitting the surrogate model
    '''
    parser = ArgumentParser()

    parser.add_argument('--surrogate', type=str, 
        choices=['botorchgp'], default='botorchgp', 
        help='type of the surrogate model')
    parser.add_argument('--mean-sample', default=False, action='store_true', 
        help='use mean sample when sampling objective functions')
    parser.add_argument('--alpha', type=float, default=alpha,
        help='VaR parameter')
    parser.add_argument('--n-w', type=int, default=n_w,
        help='number of samples for mVaR calculation')

    args, _ = parser.parse_known_args(args)
    return args


def get_acquisition_args(args=None):
    '''
    Arguments for acquisition function
    '''
    parser = ArgumentParser()

    parser.add_argument('--acquisition', type=str,  
        choices=['identity'], default='identity', 
        help='type of the acquisition function')

    args, _ = parser.parse_known_args(args)
    return args


def get_solver_args(args=None):
    '''
    Arguments for multi-objective solver
    '''
    parser = ArgumentParser()

    # general solver
    parser.add_argument('--solver', type=str, 
        choices=['nsga2', 'moead', 'discovery', 'psl','qnehvi', 'qehvi'], default='nsga2', 
        help='type of the multiobjective solver')
    parser.add_argument('--pop-size', type=int, default=500, 
        help='population size')
    parser.add_argument('--n-gen', type=int, default=100, 
        help='number of generations')
    parser.add_argument('--pop-init-method', type=str, 
        choices=['nds', 'random', 'lhs'], default='nds', 
        help='method to init population')
    parser.add_argument('--n-process', type=int, default=1,
        help='number of processes to be used for parallelization')
    parser.add_argument('--alpha', type=float, default=alpha,
        help='VaR parameter')
    parser.add_argument('--n-w', type=int, default=n_w,
        help='number of samples for mVaR calculation')
    parser.add_argument('--batch-size', type=int, default=batch_size,
        help='size of the selected batch in one iteration')

    args, _ = parser.parse_known_args(args)
    return args


def get_selection_args(args=None):
    '''
    Arguments for sample selection
    '''
    parser = ArgumentParser()

    parser.add_argument('--selection', type=str, default='hvi', 
        help='type of selection method for new batch')
    parser.add_argument('--batch-size', type=int, default=batch_size,
        help='size of the selected batch in one iteration')

    args, _ = parser.parse_known_args(args)
    return args

def get_problem_args(args=None):
    '''
    Arguments for problem
    '''
    parser = ArgumentParser()

    #sigma, repeat_eval
    parser.add_argument('--sigma', type=float, default=0.5,
        help='noise level')
    parser.add_argument('--repeat-eval', type=int, default=10,
        help='number of evaluations for each sample')
    parser.add_argument('--alpha', type=float, default=alpha,
        help='VaR parameter')
    
    args, _ = parser.parse_known_args(args)
    return args

def extract_args(args):
    
    parser = ArgumentParser()
    parser.add_argument('--args-path', type=str, default=None,
        help='used for directly loading arguments from path of argument file')
    args_p, _ = parser.parse_known_args(args)

    
    if args_p.args_path is None:

        general_args = get_general_args(args)
        surroagte_args = get_surroagte_args(args)
        acquisition_args = get_acquisition_args(args)
        solver_args = get_solver_args(args)
        selection_args = get_selection_args(args)
        problem_args = get_problem_args(args)

        framework_args = {
            'surrogate': vars(surroagte_args),
            'acquisition': vars(acquisition_args),
            'solver': vars(solver_args),
            'selection': vars(selection_args),
            'problem_args': vars(problem_args)
        }

    else:
        
        with open(args.args_path, 'r') as f:
            all_args = yaml.load(f)
        
        general_args = Namespace(**all_args['general'])
        framework_args = all_args.copy()
        framework_args.pop('general')

    return general_args, framework_args


def get_args():
    '''
    Get arguments from all components
    You can specify args-path argument to directly load arguments from specified yaml file
    '''
    parser = ArgumentParser()
    parser.add_argument('--args-path', type=str, default=None,
        help='used for directly loading arguments from path of argument file')
    args, _ = parser.parse_known_args()

    if args.args_path is None:

        general_args = get_general_args()
        surroagte_args = get_surroagte_args()
        acquisition_args = get_acquisition_args()
        solver_args = get_solver_args()
        selection_args = get_selection_args()
        problem_args = get_problem_args()

        framework_args = {
            'surrogate': vars(surroagte_args),
            'acquisition': vars(acquisition_args),
            'solver': vars(solver_args),
            'selection': vars(selection_args),
            'problem_args': vars(problem_args)
        }

    else:
        
        with open(args.args_path, 'r') as f:
            all_args = yaml.load(f)
        
        general_args = Namespace(**all_args['general'])
        framework_args = all_args.copy()
        framework_args.pop('general')

    return general_args, framework_args
