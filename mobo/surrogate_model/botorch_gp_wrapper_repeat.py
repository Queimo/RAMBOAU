import torch

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

from botorch.models.gp_regression import (
    SingleTaskGP,
)

from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize, Log
from botorch.models.transforms.input import InputPerturbation
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.acquisition.objective import ExpectationPosteriorTransform

from mobo.surrogate_model.botorch_gp_wrapper import BoTorchSurrogateModel


import gpytorch
import torch

from .botorch_helper import ZeroKernel
from pathlib import Path


class BoTorchSurrogateModelReapeat(BoTorchSurrogateModel):
    """
    Gaussian process
    """

    def __init__(self, n_var, n_obj, **kwargs):
        super().__init__(n_var, n_obj)
        self.n_w = kwargs["n_w"]
        self.alpha = kwargs["alpha"]
        self.noise_model = None
        self.input_transform = InputPerturbation(
            torch.zeros((self.n_w, self.n_var), **tkwargs)
        )

    def save(self, path):
        self.state_dict = self.bo_model.state_dict()
        torch.save(self.state_dict, Path(path) / "state_dict.pt")
        self.state_dict_noise = self.noise_model.state_dict()
        torch.save(self.state_dict_noise, Path(path) / "state_dict_noise.pt")
        
        
    def initialize_model(self, train_x, train_y, train_rho=None, state_dict=None):
        # define models for objective and constraint
        train_y_mean = -train_y  # negative because botorch assumes maximization

        models = []
        for i in range(train_y_mean.shape[1]):
            model = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y_mean[..., i : i + 1],
                input_transform=self.input_transform,
                outcome_transform=Standardize(m=1),
            )

            models.append(model)
            

        model = ModelListGP(*models)
        
        noise_model_list = self.initialize_noise_model(train_x.clone(), train_rho.clone())
        self.noise_model = ModelListGP(*noise_model_list)
        
        models_for_mll = models + noise_model_list
        mll_model = ModelListGP(*models_for_mll)
        
        
        if state_dict is not None:
            model.load_state_dict(state_dict)
        
        mll = SumMarginalLogLikelihood(mll_model.likelihood, mll_model)
        
        return mll, model

    def initialize_noise_model(self, train_x, train_var):
        train_var_mean = train_var + 1e-6
        # train_y_var = torch.tensor(train_rho, **tkwargs) + 1e-6

        models = []
        for i in range(train_var_mean.shape[1]):
            model = SingleTaskGP(
                train_X=train_x,
                train_Y=train_var_mean[..., i : i + 1],
                # likelihood=GaussianLikelihood(
                #     noise_prior=SmoothedBoxPrior(-3, 5, 0.5, transform=torch.log),
                #     noise_constraint=GreaterThan(
                #         MIN_INFERRED_NOISE_LEVEL, transform=None, initial_value=1.0
                #     ),
                # ),
                input_transform=self.input_transform,
                outcome_transform=Log(),
            )

            models.append(model)
        
        return models

    def evaluate(
        self,
        X,
        std=False,
        noise=False,
        calc_gradient=False,
        calc_hessian=False,
    ):
        X = torch.tensor(X).to(**tkwargs)

        F, dF, hF = None, None, None  # mean
        S, dS, hS = None, None, None  # std
        rho_F, drho_F = None, None  # noise mean
        rho_S, drho_S = None, None  # noise std
        mvar_F = None

        model = self.bo_model
        
        
        with torch.no_grad():
            post = model.posterior(
                X, posterior_transform=ExpectationPosteriorTransform(n_w=self.n_w)
            )
            # negative because botorch assumes maximization (undo previous negative)
            F = -post.mean.detach().cpu().numpy()
            S = post.variance.sqrt().detach().cpu().numpy()

            if noise:
                noise_post = self.noise_model.posterior(X)
                rho_F = noise_post.mean.detach().cpu().numpy()[::self.n_w]
                rho_S = noise_post.variance.sqrt().detach().cpu().numpy()[::self.n_w]
                
                # if isinstance(model.likelihood, LikelihoodList):
                #     rho_F = np.zeros_like(F)
                #     rho_S = np.zeros_like(S)
                #     for i, likelihood in enumerate(model.likelihood.likelihoods):
                #         if hasattr(likelihood.noise_covar, "noise_model"):
                #             rho_post = likelihood.noise_covar.noise_model.posterior(X)
                #             rho_F_i = rho_post.mean.detach().cpu().numpy()[::self.n_w]
                #             rho_S_i = rho_post.variance.sqrt().detach().cpu().numpy()[::self.n_w]
                #             if F.shape[1] == 1:
                #                 rho_F = rho_F_i
                #                 rho_S = rho_S_i
                #             else:
                #                 rho_F[:, i] = rho_F_i.squeeze(-1)
                #                 rho_S[:, i] = rho_S_i.squeeze(-1)
                # else:
                #     rho_post = model.likelihood.noise_covar.noise_model.posterior(X)
                #     rho_F = rho_post.mean.detach().cpu().numpy()[::self.n_w, :]
                #     rho_S = rho_post.variance.sqrt().detach().cpu().numpy()[::self.n_w, :]
                
                # rho_F = model.posterior(X, posterior_transform=ExpectationPosteriorTransform(n_w=self.n_w), observation_noise=True).variance.squeeze(-1).detach().cpu().numpy()
                # rho_F= (rho1-S**2).clip(min=0)

        # #simplest 2d --> 2d test problem
        # def f(X):
        #     return torch.stack([X[:, 0]**2 + 0.1*X[:, 1]**2 , -X[:, 1]**2 -0.1*(X[:, 0]**2)]).T
        # X_toy = torch.tensor([[0.5, 0.5], [1., 1.], [2., 2.]], requires_grad=True)
        # jacobian_mean = torch.autograd.functional.jacobian(f, X_toy)
        # # goal 3 x 2 x 2
        # jac_batch = jacobian_mean.diagonal(dim1=0,dim2=2).transpose(0,-1).transpose(1,2).numpy()

        if calc_gradient:
            jac_F = torch.autograd.functional.jacobian(
                lambda x: -model.posterior(
                    x, posterior_transform=ExpectationPosteriorTransform(n_w=self.n_w)
                ).mean.T,
                X,
            )
            dF = (
                jac_F.diagonal(dim1=0, dim2=2)
                .transpose(0, -1)
                .transpose(1, 2)
                .detach()
                .cpu()
                .numpy()
            )

            if std:
                jac_S = torch.autograd.functional.jacobian(
                    lambda x: model.posterior(x).variance.sqrt().T, X
                )
                dS = (
                    jac_S.diagonal(dim1=0, dim2=2)
                    .transpose(0, -1)
                    .transpose(1, 2)
                    .detach()
                    .cpu()
                    .numpy()
                )

        out = {
            "F": F,
            "dF": dF,
            "hF": hF,
            "S": S,
            "dS": dS,
            "hS": hS,
            "rho_F": rho_F,
            "drho_F": drho_F,
            "rho_S": rho_S,
            "drho_S": drho_S,
            "mvar_F": mvar_F,
        }

        return out

class BoTorchSurrogateModelReapeatMean(BoTorchSurrogateModelReapeat):

    def __init__(self, n_var, n_obj, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
    
    def initialize_model(self, train_x, train_y, train_rho):
        # define models for objective and constraint
        mll, model_list_GP = super().initialize_model(train_x, train_y[...,:-1], train_rho[...,:-1])
        
        class LinearMean(gpytorch.means.Mean):
            def __init__(self):        
                super(LinearMean, self).__init__()
                
            def forward(self, x):
                """Your forward method."""
                # Stoichiometric balance
                c_ohzn = x[..., 0]
                c_zn = x[..., 1]
                q_AC = x[..., 2]
                Q_AC = q_AC * 6 + 4
                C_Zn = c_zn * 0.9 + 0.1
                C_OHZn = c_ohzn * 3. + 0.5 #unnormalize based on bounds
                C_OH = C_OHZn * C_Zn
                C_ZnO = torch.min(C_Zn, 0.5 * C_OH)
                N_ZnO = C_ZnO * Q_AC
                return N_ZnO
        
        # should not have any effect
        train_y_mean = -train_y  # negative because botorch assumes maximization
        train_y_var = train_rho + 1e-6
        
        models = [*model_list_GP.models]
        
        model_mean = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y_mean[..., -1:],
            train_Yvar=train_y_var[..., -1:]*0.0,
            likelihood= gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-10)),
            input_transform=self.input_transform,
            mean_module=LinearMean(),
            covar_module=ZeroKernel(),
        )
        model_mean.likelihood.noise_covar.noise = 1e-10
        
        models.append(model_mean)
        model = ModelListGP(*models)
        
        noise_model_list = [*self.noise_model.models]
        
        zero_noise_model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y_var[..., -1:]*0.0,
            train_Yvar=train_y_var[..., -1:]*0.0,
            likelihood= gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-10)),
            input_transform=self.input_transform,
            covar_module=ZeroKernel(),
        )
        
        self.noise_model = ModelListGP(*noise_model_list, zero_noise_model)
        
        return mll, model

        