import os
import uuid
# os.environ["OMP_NUM_THREADS"] = "1"  # speed up
import numpy as np
from problems.common import build_problem
from mobo.algorithms import get_algorithm
from visualization.data_export import DataExport
from utils import save_args
from utils import RefPoint
import torch
import gc
from mobo.mobo import MOBO

import numpy as np
from mobo.surrogate_problem import SurrogateProblem
from mobo.utils import (
    Timer,
    find_pareto_front,
    calc_hypervolume,
)
from mobo.factory import init_from_config
from mobo.transformation import StandardTransform
import wandb

from mobo.algorithms import RAqNEHVI, qNEHVI, RAqNEIRS, RAqLogNEHVI, RAqNEIRSdet, qNEHVIdet

# algo_name = 'raqlognehvidet'
# algo_name = 'raqnehvi'
algo_name = 'raqneirsdet'
# algo_name = 'raqneirs'
ALGO = get_algorithm(algo_name)

class MOBOEXP(ALGO):
    """
    Base class of algorithm framework, inherit this class with different configs to create new algorithm classes
    """

    def __init__(self, problem, n_iter, ref_point_handler, framework_args, batch_size=6):
        
        self.df = problem.df_mean_var
        self.problem = problem
        self.batch_size = batch_size
        
        super().__init__(problem, n_iter, ref_point_handler, framework_args)

    def step(self):
        
        timer = Timer()

        # data normalization
        self.transformation.fit(self.X, self.Y)
        X = self.transformation.do(self.X)
        Y, rho = self.Y, self.rho
        # X, Y, rho = self.transformation.do(self.X, self.Y, self.rho)

        # # build surrogate models
        self.surrogate_model.fit(X, Y, rho)
        timer.log("Surrogate model fitted")

        # define acquisition functions
        self.acquisition.fit(X, Y)

        # solve surrogate problem
        surr_problem = SurrogateProblem(
            self.real_problem,
            self.surrogate_model,
            self.acquisition,
            self.transformation,
        )
        if self.sample_num >= len(self.df):

            solution = self.solver.solve(surr_problem, X, Y, rho)
            timer.log("Surrogate problem solved")
            # batch point selection
            self.selection.fit(X, Y)
            X_next, self.info = self.selection.select(
                solution, self.surrogate_model, self.status, self.transformation
            )
            timer.log("Next sample batch selected")

            # update dataset
            # Y_next, rho_next = self.real_problem.evaluate(X_next, return_values_of=['F', 'rho'])
            # Don't need to evaluate real problem because we don't have the data yet
            # evaluate prediction of X_next on surrogate model
            val = self.surrogate_model.evaluate(self.transformation.do(x=X_next), std=True, noise=True)
            # Y_next_pred_mean, rho_next = self.transformation.undo(y=val['F'], rho=val['rho_F'])
            Y_next_pred_mean = val['F']
            rho_next_pred = val['rho_F']
            rho_next = rho_next_pred
            Y_next = Y_next_pred_mean
            Y_next_pred_std = val['S']
            acquisition, _, _ = self.acquisition.evaluate(val)
        
        else:
            # only solve for current pareto front with NSGA of gp
            solution = self.solver.nsga2_solve(surr_problem, X, Y)
            self.solver.solution = solution
            timer.log("Surrogate problem solved")
            X_next = self.df.iloc[self.sample_num:self.sample_num+self.batch_size][self.problem.var_cols].values
            Y_next = -1 * self.df.iloc[self.sample_num:self.sample_num+self.batch_size][self.problem.obj_cols].values # we assume minimzation
            rho_next = self.df.iloc[self.sample_num:self.sample_num+self.batch_size][[col.replace("_mean","_var") for col in self.problem.obj_cols]].values
            
            val = self.surrogate_model.evaluate(self.transformation.do(x=X_next), std=True, noise=True)
            # Y_next_pred_mean, rho_next = self.transformation.undo(y=val['F'], rho=val['rho_F'])
            Y_next_pred_mean = val['F']
            rho_next_pred = val['rho_F']
            Y_next_pred_std = val['S']
            acquisition, _, _ = self.acquisition.evaluate(val)

        
        self._update_status(X_next, Y_next, rho=rho_next)
        timer.log("New samples evaluated")

        # statistics
        self.global_timer.log("Total runtime", reset=False)
        print(f"Total evaluations: {self.sample_num}, mVaR hypervolume: {self.status['mvar_hv']:.4f}, hypervolume: {self.status['hv']:.4f}")

        # return new data iteration by iteration
        return X_next, Y_next, rho_next, Y_next_pred_mean, Y_next_pred_std, acquisition


def run_experiment(args, framework_args):
    # load arguments

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # build problem, get initial samples
    problem, true_pfront, X_init, Y_init, rho_init = build_problem(
        args.problem, args.n_var, args.n_obj, args.n_init_sample, args.n_process
    )
    
    dataset = problem.df_mean_var

    assert args.n_init_sample <= len(dataset), "Initial sample size is larger than the available data"
    assert args.n_iter*args.batch_size + args.n_init_sample == len(dataset) + args.batch_size, f"Total number of iterations and batch size is not equal to the total number of data points, {args.n_iter*args.batch_size + args.n_init_sample} != {len(dataset) + args.batch_size}"
    
    if args.only_recorded_data:
        args.n_iter -= 1
    
    args.n_var, args.n_obj = problem.n_var, problem.n_obj

    ref_point_handler = RefPoint(
        args.problem, args.n_var, args.n_obj, n_init_sample=args.n_init_sample
    )

    args.ref_point = ref_point_handler.get_ref_point(is_botorch=False)

    # initialize optimizer
    
    optimizer = MOBOEXP(
        problem, args.n_iter, ref_point_handler, framework_args, batch_size=args.batch_size
    )
    
    
    # save arguments & setup logger
    save_args(args, framework_args)
    print(problem, optimizer, sep="\n")

    # initialize data exporter
    exporter = DataExport(optimizer, X_init, Y_init, rho_init, args)

    # optimization
    solution = optimizer.solve(X_init, Y_init, rho_init)

    # export true Pareto front to csv
    if true_pfront is not None:
        exporter.write_truefront_csv(true_pfront)

    for step in range(args.n_iter):
        # get new design samples and corresponding performance
        X_next, Y_next, rho_next, Y_next_pred_mean, Y_next_pred_std, acq = next(
            solution
        )
        # update & export current status to csv
        exporter.update(X_next, Y_next, Y_next_pred_mean, Y_next_pred_std, acq, rho_next)

        gc.collect()

        exporter.write_csvs()
        exporter.save_psmodel()
    
    

if __name__ == "__main__":
    from arguments import get_args

    args, framework_args = get_args()
    
    args.algo = algo_name
    args.problem = 'exp4d'
    args.n_iter = 12
    args.only_recorded_data = False
    args.n_init_sample = 12
    args.batch_size = 2
    args.subfolder = 'unroll_new'
    framework_args["solver"]["batch_size"] = args.batch_size
    framework_args["selection"]["batch_size"] = args.batch_size

    run_experiment(args, framework_args)
