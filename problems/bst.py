import numpy as np
import sys
sys.path.append('..')
sys.path.append('.')
try :
    from .problem import Problem, RiskyProblem
except:
    from problem import Problem, RiskyProblem
from mobo.utils import calculate_var, find_pareto_front


class BST(RiskyProblem):

    def __init__(self, sigma, repeat_eval, alpha=0.9):
        
        self.sigma = sigma
        self.repeat_eval = repeat_eval
        self.bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.dim = 2
        self.num_objectives = 2
        self.alpha = alpha
        self.ref_point = None
        
        super().__init__(
            n_var=self.dim, 
            n_obj=self.num_objectives, 
            n_constr=0,
            xl=self.bounds[0,:],
            xu=self.bounds[1,:],
        )
    
    
    def evaluate_repeat_mvar(self, x: np.array) -> np.array:
        y_true = self.f(x)
        variance = self.get_noise_var(x)
        mvar = calculate_var(y_true, variance=variance, alpha=self.alpha)
        mvar = np.expand_dims(mvar,-1)
        # print("mvar calculated")
        return mvar
    
    def pareto_front(self, n_pareto_points=500, alphas=[0.5, 0.9, 0.99], return_x=False):
        try:
            from common import generate_initial_samples
        except:
             from .common import generate_initial_samples
             
        from mobo.solver import NSGA2Solver
        from arguments import get_solver_args
         
        solver_args = vars(get_solver_args())
        solver_args['pop_size'] = n_pareto_points
        
        Ys = []
        Xs = []

        for alpha in alphas:
            prob = self.__class__(sigma=self.sigma, repeat_eval=1, alpha=alpha)
            prob.evaluate_repeat = prob.evaluate_repeat_mvar
            X_init, Y_init, rho_init = generate_initial_samples(prob, n_pareto_points)
            solver = NSGA2Solver(**solver_args)

            solution = solver.solve(prob, X_init, Y_init)
            
            Y_pareto, idx_pareto = find_pareto_front(solution['y'], return_index=True)
            X_pareto = solution['x'][idx_pareto]
            Ys.append(Y_pareto)
            Xs.append(X_pareto)
        
        if return_x:
            return Ys, Xs
        else:
            return Ys
    
    def evaluate_repeat(self, x: np.array) -> np.array:
        y_true = self.f(x)
        sigmas = self.get_noise_var(x)
        y_true = np.stack([y_true] * self.repeat_eval, axis=-1)
        y = y_true + np.expand_dims(sigmas, -1) * np.random.randn(*y_true.shape)
        return y

    def get_domain(self):
        return self.bounds
    
    def styblinski_tang_function(self, x1, x2):
        x1 = 10 * x1 - 5
        x2 = 10 * x2 - 5
        return 0.5 * ((x1**4 - 16 * x1**2 + 5 * x1) + (x2**4 - 16 * x2**2 + 5 * x2)) / 250

    def brannin_function(self, x1, x2):
        x1 = 15 * x1 - 5
        x2 = 15 * x2
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        return a * ((x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s) / 300

    def f(self, X):
        x1, x2 = X[:, 0], X[:, 1]
        y1 = self.brannin_function(x1, x2)
        y2 = self.styblinski_tang_function(x1, x2)
        
        return np.stack([y1, y2], axis=-1)
        # return np.stack([300*(x1-x2)**3, 200*(x2-x1)**3], axis=-1)
        
    def sigmoid(self, x):
        """ Sigmoid function for scaling. """
        return 1 / (1 + np.exp(-x))

    def get_noise_var(self, X):
        pass

    def __str__(self):
        return super().__str__() + f"\nSigma: {self.sigma}, Repeat eval: {self.repeat_eval}, Alpha: {self.alpha}"

class BSTVert(BST):
    def __init__(self, sigma, repeat_eval, alpha=0.9):
        
        super().__init__(sigma, repeat_eval, alpha)
        self.sigma = sigma
        self.repeat_eval = repeat_eval
        self.alpha = alpha
        self.ref_point = [0.07, 0.04]
    
    def get_noise_var(self, X):
        # return self.sigma * self.sigmoid(5*(X[:,1]-0.5))
        x1, x2 = X[:, 0], X[:, 1]
        noise_factor = self.sigmoid(5 * (x2 - 0.5))
        rho1 = self.sigma * noise_factor
        rho2 = self.sigma * noise_factor
        return np.stack([rho1, rho2], axis=-1)

class BSTHorz(BST):
    def __init__(self, sigma, repeat_eval, alpha=0.9):
        
        super().__init__(sigma, repeat_eval, alpha)
        self.sigma = sigma
        self.repeat_eval = repeat_eval
        self.alpha = alpha
        self.ref_point = [0.12, 0.23]
        
    def get_noise_var(self, X):
        
        x1, x2 = X[:, 0], X[:, 1]
        # noise_factor = self.sigmoid(20 * (x1 - 0.6)) * self.sigmoid(20 * (-x2 + 0.4))
        noise_factor = self.sigmoid(20 * (-x1 + 0.5))
        rho1 = self.sigma * noise_factor
        rho2 = self.sigma * noise_factor
        
        return np.stack([rho1, rho2], axis=-1)

class BSTDiag(BST):
    def __init__(self, sigma, repeat_eval, alpha=0.9):
        
        super().__init__(sigma, repeat_eval, alpha)
        self.sigma = sigma
        self.repeat_eval = repeat_eval
        self.alpha = alpha
        self.ref_point = [0.48, 0.52]

    def get_noise_var(self, X):
        
        x1, x2 = X[:, 0], X[:, 1]
        x = np.sum(X, axis=-1)
        # noise_factor = self.sigmoid(20 * (x1 - 0.6)) * self.sigmoid(20 * (-x2 + 0.4))
        noise_factor = self.sigmoid(20 * (-x + 1))
        rho1 = self.sigma * noise_factor
        rho2 = self.sigma * noise_factor
        
        return np.stack([rho1, rho2], axis=-1)

if __name__ == "__main__":

    prob = BSTDiag(sigma=.5, repeat_eval=1)

    alphas = [0.5, 0.9, 0.99]
    alphas = np.linspace(0.5, 0.99, 10)
    Ys = prob.pareto_front(alphas=alphas)
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    for i, Y_paretos in enumerate(Ys):
        fig.add_trace(go.Scatter(x=Ys[i][:,0], y=Ys[i][:,1], mode='markers', name=f'alpha={alphas[i]}'))
    
    
    
    fig.show()
    
    # The plot should show the pareto front, the lower bound of the pareto front, and the upper bound of the pareto front.
    # The pareto front should be between the lower and upper bounds.