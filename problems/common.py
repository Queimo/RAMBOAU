import numpy as np
from problems import *
from baselines.external import lhs
from botorch.utils.sampling import draw_sobol_samples
import torch


def get_problem_options():
    problems_dict = {
    'bstvert': BSTVert,
    'bsthorz': BSTHorz,
    'bstdiag': BSTDiag,
    'exp4d': Experiment4D,
}

    return problems_dict


def get_problem(name, *args, d={}, **kwargs):
    return get_problem_options()[name](*args, **d, **kwargs)



def generate_initial_samples(problem, n_sample):
    '''
    Generate feasible initial samples.
    Input:
        problem: the optimization problem
        n_sample: number of initial samples
    Output:
        X, Y: initial samples (design parameters, performances)
    '''
    X_feasible = np.zeros((0, problem.n_var))
    Y_feasible = np.zeros((0, problem.n_obj))
    rho_feasible = np.zeros((0, problem.n_obj))

    # NOTE: when it's really hard to get feasible samples, the program hangs here
    while len(X_feasible) < n_sample:
        X = draw_sobol_samples(bounds=torch.tensor(problem.bounds), n=n_sample, q=1).squeeze().numpy()
        Y, feasible, rho = problem.evaluate(X, return_values_of=['F', 'feasible', 'rho'])
        feasible = feasible.flatten()
        X_feasible = np.vstack([X_feasible, X[feasible]])
        Y_feasible = np.vstack([Y_feasible, Y[feasible]])
        if rho is not None:
            rho_feasible = np.vstack([rho_feasible, rho[feasible]]) 
        else:
            rho_feasible = None
        
    
    indices = np.random.permutation(np.arange(len(X_feasible)))[:n_sample]
    X, Y, rho = X_feasible[indices], Y_feasible[indices], rho_feasible[indices] if rho_feasible is not None else None
    
    
    return X, Y, rho


def build_problem(name, n_var, n_obj, n_init_sample, n_process=1, **problem_kwargs):
    '''
    Build optimization problem from name, get initial samples
    Input:
        name: name of the problem (supports ZDT1-6, DTLZ1-7)
        n_var: number of design variables
        n_obj: number of objectives
        n_init_sample: number of initial samples
        n_process: number of parallel processes
    Output:
        problem: the optimization problem
        X_init, Y_init: initial samples
        pareto_front: the true pareto front of the problem (if defined, otherwise None)
    '''
    problem = get_problem(name, **problem_kwargs)

    pareto_front = problem.pareto_front()
    # try:
    #     pareto_front = problem.pareto_front()
    # except Exception as e:
    #     print('no true pareto front defined for this problem!')
    #     print(e)
    #     pareto_front = None

    # get initial samples
    if 'exp' in name:
        X_init = problem.X[:n_init_sample]
        Y_init = problem.Y[:n_init_sample]
        rho_init = problem.rho[:n_init_sample]
    else:
        X_init, Y_init, rho_init = generate_initial_samples(problem, n_init_sample)
    
    return problem, pareto_front, X_init, Y_init, rho_init