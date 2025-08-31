import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from argparse import ArgumentParser
from problems.common import build_problem

import os, sys
import yaml

import torch
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from mobo.utils import calculate_var, find_pareto_front

class RefPoint:
    
    ref_point_botroch = None
    ref_point_pymoo = None
    
    def __init__(self, Y_init, rho_init, alpha, problem=None):
        if problem is not None:
            if problem.ref_point is not None:
                self.ref_point_botroch = (-np.array(problem.ref_point)).tolist()
                self.ref_point_pymoo = problem.ref_point
                return
            
        mvar = calculate_var(Y_init, variance=rho_init, alpha=self.solver.alpha)
        mvar_pfront, mvar_pidx = find_pareto_front(mvar, return_index=True)
    
        self.ref_point_botroch = infer_reference_point(torch.tensor(-mvar_pfront)).numpy().tolist()
        # self.ref_point_botroch = np.max(-Y_init, axis=0).tolist()
        # self.ref_point_pymoo= np.max(Y_init, axis=0).tolist()
        self.ref_point_pymoo = (-infer_reference_point(torch.tensor(-mvar_pfront)).numpy()).tolist()
        print(self)

    def get_ref_point(self, is_botorch=False):

        if is_botorch:
            return self.ref_point_botroch
        else:
            return self.ref_point_pymoo
        
    def __str__(self):
        return f'Ref point: {self.ref_point_pymoo} \n Ref point (botorch): {self.ref_point_botroch}'
    
    def __repr__(self): 
        return self.__str__()



def get_result_dir(args):
    '''
    Get directory of result location (result/problem/subfolder/algo/seed/)
    '''
    top_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    exp_name = '' if args.exp_name is None else '-' + args.exp_name
    algo_name = args.algo + exp_name
    result_dir = os.path.join(top_dir, args.problem, args.subfolder, algo_name, str(args.seed))
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_args(general_args, framework_args):
    '''
    Save arguments to yaml file
    '''
    all_args = {'general': vars(general_args)}
    all_args.update(framework_args)

    result_dir = get_result_dir(general_args)
    args_path = os.path.join(result_dir, 'args.yml')

    os.makedirs(os.path.dirname(args_path), exist_ok=True)
    with open(args_path, 'w') as f:
        yaml.dump(all_args, f, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--n-var', type=int, default=2)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--n-init-sample', type=int, default=500)
    args = parser.parse_args()
    np.random.seed(0)
    _, _, _, Y_init, rho_init = build_problem(args.problem, args.n_var, args.n_obj, args.n_init_sample)

    ref_point_handler = RefPoint(Y_init)

    print(ref_point_handler)