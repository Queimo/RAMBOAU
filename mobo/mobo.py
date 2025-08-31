import numpy as np
from .surrogate_problem import SurrogateProblem
from .utils import (
    Timer,
    find_pareto_front,
    calc_hypervolume,
    calculate_var
)
from .factory import init_from_config
from .transformation import StandardTransform

"""
Main algorithm framework for Multi-Objective Bayesian Optimization
"""


class MOBO:
    """
    Base class of algorithm framework, inherit this class with different configs to create new algorithm classes
    """

    config = {}

    def __init__(self, problem, n_iter, ref_point_handler, framework_args):
        """
        Input:
            problem: the original / real optimization problem
            n_iter: number of iterations to optimize
            ref_point: reference point for hypervolume calculation
            framework_args: arguments to initialize each component of the framework
        """
        self.real_problem = problem
        self.n_var, self.n_obj = problem.n_var, problem.n_obj
        self.n_iter = n_iter
        self.ref_point_handler = ref_point_handler

        bounds = np.array([problem.xl, problem.xu])
        self.transformation = StandardTransform(
            bounds
        )  # data normalization for surrogate model fitting
        self.bounds = bounds

        # framework components
        framework_args["surrogate"]["n_var"] = self.n_var  # for surrogate fitting
        framework_args["surrogate"]["n_obj"] = self.n_obj  # for surroagte fitting
        framework_args["solver"]["n_obj"] = self.n_obj  # for MOEA/D-EGO
        framework = init_from_config(self.config, framework_args)

        self.surrogate_model = framework["surrogate"]  # surrogate model
        self.acquisition = framework["acquisition"]  # acquisition function
        self.solver = framework[
            "solver"
        ]  # multi-objective solver for finding the paretofront
        self.selection = framework[
            "selection"
        ]  # selection method for choosing new (batch of) samples to evaluate on real problem

        # to keep track of data and pareto information (current status of algorithm)

        self.sample_num = 0
        self.status = {
            'pset': None,
            'pfront': None,
            'hv': None,
            'ref_point': self.ref_point_handler.get_ref_point(is_botorch=False),
            'mvar_pset': None,
            'mvar_pfront': None,
            'mvar_hv': None,
        }

        # other component-specific information that needs to be stored or exported
        self.info = None
        self.global_timer = None

    def _update_status(self, X, Y, rho=None):
        '''
        Update the status of algorithm from data
        '''
        if self.sample_num == 0:
            self.X = X
            self.Y = Y
            self.rho = rho
        else:
            self.X = np.vstack([self.X, X])
            self.Y = np.vstack([self.Y, Y])
            self.rho = np.vstack([self.rho, rho]) if rho is not None else None
        self.sample_num += len(X)

        self.status['pfront'], pfront_idx = find_pareto_front(self.Y, return_index=True)
        self.status['pset'] = self.X[pfront_idx]
        self.status['hv'] = calc_hypervolume(self.status['pfront'], self.ref_point_handler.get_ref_point(is_botorch=False))
        
        #MVaR Calculation
        
        #compute MVaR HV
        mvar = calculate_var(self.Y, variance=self.rho, alpha=self.solver.alpha)
        mvar_pfront, mvar_pidx = find_pareto_front(mvar, return_index=True)
        mvar_pset = self.X[mvar_pidx]
        mvar_hv_value = calc_hypervolume(mvar_pfront, ref_point=self.ref_point_handler.get_ref_point(is_botorch=False))
    
        self.status['mvar'] = mvar       
        self.status['mvar_pfront'] = mvar_pfront
        self.status['mvar_pset'] = mvar_pset
        self.status['mvar_hv'] = mvar_hv_value
        
        # print('Current hypervolume: %.4f' % self.status['hv'])
    
    def step(self):
        
        timer = Timer()

        # data normalization
        self.transformation.fit(self.X, self.Y)
        X = self.transformation.do(self.X)
        Y, rho = self.Y, self.rho

        # build surrogate models
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
        solution = self.solver.solve(surr_problem, X, Y, rho)
        timer.log("Surrogate problem solved")

        # batch point selection
        self.selection.fit(X, Y)
        X_next, self.info = self.selection.select(
            solution, self.surrogate_model, self.status, self.transformation
        )
        timer.log("Next sample batch selected")

        # update dataset
        Y_next, rho_next = self.real_problem.evaluate(X_next, return_values_of=['F', 'rho'])
        # evaluate prediction of X_next on surrogate model
        val = self.surrogate_model.evaluate(self.transformation.do(x=X_next), std=True)
        Y_next_pred_mean = val['F']
        Y_next_pred_std = val['S']
        acquisition, _, _ = self.acquisition.evaluate(val)

        
        if self.real_problem.n_constr > 0:
            Y_next = Y_next[0]
            rho_next = rho_next[0]
        self._update_status(X_next, Y_next, rho=rho_next)
        timer.log("New samples evaluated")

        # statistics
        self.global_timer.log("Total runtime", reset=False)
        print(f"Total evaluations: {self.sample_num}, mVaR hypervolume: {self.status['mvar_hv']:.4f}, hypervolume: {self.status['hv']:.4f}")

        # return new data iteration by iteration
        return X_next, Y_next, rho_next, Y_next_pred_mean, Y_next_pred_std, acquisition

    def init_solve(self, X_init, Y_init, rho_init=None):
        self.selection.set_ref_point(self.ref_point_handler.get_ref_point(is_botorch=False))
        self.solver.set_ref_point(self.ref_point_handler.get_ref_point(is_botorch=True)) # botorch need a different ref point because it assumes maximization and boxdecomposition fails otherwise

        self._update_status(X_init, Y_init, rho_init)

        self.global_timer = Timer()
        

    def solve(self, X_init, Y_init, rho_init=None):
        """
        Solve the real multi-objective problem from initial data (X_init, Y_init)
        """
        self.init_solve(X_init, Y_init, rho_init)
        
        for i in range(self.n_iter):
            print("========== Iteration %d ==========" % i)
            
            
            X_next, Y_next, rho_next, Y_next_pred_mean, Y_next_pred_std, acquisition = self.step()
            
            yield X_next, Y_next, rho_next, Y_next_pred_mean, Y_next_pred_std, acquisition
            
        

    def __str__(self):
        return (
            "========== Framework Description ==========\n"
            + f"# algorithm: {self.__class__.__name__}\n"
            + f"# surrogate: {self.surrogate_model.__class__.__name__}\n"
            + f"# acquisition: {self.acquisition.__class__.__name__}\n"
            + f"# solver: {self.solver.__class__.__name__}\n"
            + f"# selection: {self.selection.__class__.__name__}\n"
        )
