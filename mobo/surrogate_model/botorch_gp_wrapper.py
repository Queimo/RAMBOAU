import numpy as np
import torch

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

from botorch.models.gp_regression import (
    SingleTaskGP,
)

from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import LikelihoodList
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.models.transforms.outcome import Standardize

from mobo.surrogate_model.base import SurrogateModel

import gpytorch
import torch

import torch

import torch
import gpytorch

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
from .botorch_helper import (
    ZeroKernel,
    CustomHeteroskedasticSingleTaskGP,
)

from pathlib import Path


class BoTorchSurrogateModel(SurrogateModel):
    """
    Gaussian process
    """

    def __init__(self, n_var, n_obj, **kwargs):
        self.bo_model = None
        self.input_transform = None
        super().__init__(n_var, n_obj)
        
    def save(self, path):
        self.state_dict = self.bo_model.state_dict()
        torch.save(self.state_dict, Path(path) / "state_dict.pt")
        
        

    def _fit(self, mll, X_torch=None, Y_torch=None, rho_torch=None):

        try:
            fit_gpytorch_mll(mll, max_retries=5)
            return
        except Exception as e:
            print(e)
            print("failed to fit, retrying with torch optim...")

        mll, self.bo_model = self.initialize_model(X_torch, Y_torch, rho_torch)
        try:
            fit_gpytorch_mll_torch(mll, step_limit=2000)
        except Exception as e:
            print(e)
            print("failed to fit.")
        

    def fit(self, X, Y, rho=None):
        X_torch = torch.tensor(X).to(**tkwargs).detach()
        Y_torch = torch.tensor(Y).to(**tkwargs).detach()
        rho_torch = (
            torch.tensor(rho).to(**tkwargs).detach() if rho is not None else None
        )
        # print all statistics of rho for debugging
        print("rho statistics")
        for i in range(Y_torch.shape[1]):
            print(
                f"mean: {rho_torch[:,i].mean()}, std: {rho_torch[:,i].std()}, min: {rho_torch[:,i].min()}, max: {rho_torch[:,i].max()}"
            )

        mll, self.bo_model = self.initialize_model(X_torch, Y_torch, rho_torch)

        self._fit(mll, X_torch.clone(), Y_torch.clone(), rho_torch.clone())
        # self._fit(mll_noise, X_torch, rho_torch, torch.zeros_like(rho_torch))

    def initialize_model(self, train_x, train_y, train_rho=None, state_dict=None):
        # define models for objective and constraint
        train_y_mean = -train_y  # negative because botorch assumes maximization
        train_y_var = train_rho + 1e-10

        models = []
        for i in range(train_y_mean.shape[1]):
            model = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y_mean[..., i : i + 1],
                train_Yvar=train_y_var[..., i:i+1],
                input_transform=self.input_transform,
                outcome_transform=Standardize(m=1),
            )

            models.append(model)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

        model = ModelListGP(*models)
        
        if state_dict is not None:
            model.load_state_dict(state_dict)
        
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        
        return mll, model

    def evaluate(
        self,
        X,
        std=False,
        noise=False,
        calc_gradient=False,
        calc_hessian=False,
        calc_mvar=False,
    ):
        X = torch.tensor(X).to(**tkwargs)

        F, dF, hF = None, None, None  # mean
        S, dS, hS = None, None, None  # std
        rho_F, drho_F = None, None  # noise mean
        rho_S, drho_S = None, None  # noise std

        model = self.bo_model

        post = self.bo_model.posterior(X)
        # negative because botorch assumes maximization (undo previous negative)
        F = -post.mean.squeeze(-1).detach().cpu().numpy()
        S = post.variance.sqrt().squeeze(-1).detach().cpu().numpy()

        if noise:
            # noise_post = self.noise_model.posterior(X)
            # rho_F = noise_post.mean.squeeze(-1).detach().cpu().numpy()
            # rho_S = noise_post.variance.sqrt().squeeze(-1).detach().cpu().numpy()

            if isinstance(model.likelihood, LikelihoodList):
                rho_F = np.zeros_like(F)
                rho_S = np.zeros_like(S)
                for i, likelihood in enumerate(model.likelihood.likelihoods):
                    if hasattr(likelihood.noise_covar, "noise_model"):
                        rho_post = likelihood.noise_covar.noise_model.posterior(X)
                        rho_F_i = rho_post.mean.detach().cpu().numpy()
                        rho_S_i = rho_post.variance.sqrt().detach().cpu().numpy()
                        if F.shape[1] == 1:
                            rho_F = rho_F_i
                            rho_S = rho_S_i
                        else:
                            rho_F[:, i] = rho_F_i.squeeze(-1)
                            rho_S[:, i] = rho_S_i.squeeze(-1)
            else:
                rho_post = model.likelihood.noise_covar.noise_model.posterior(X)
                rho_F = rho_post.mean.detach().cpu().numpy()
                rho_S = rho_post.variance.sqrt().detach().cpu().numpy()
                
        if calc_gradient:
            if isinstance(model.likelihood, LikelihoodList):
                dF = np.zeros((X.shape[0], F.shape[1], X.shape[1]))
                dS = np.zeros((X.shape[0], F.shape[1], X.shape[1]))
                for i, m in enumerate(model.models):
                    jac_F = torch.autograd.functional.jacobian(
                        lambda x: -m.posterior(x).mean.T, X
                    )
                    dF[..., i] = (
                        jac_F.diagonal(dim1=0, dim2=2)
                        .transpose(0, -1)
                        .transpose(1, 2)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    if std:
                        jac_S = torch.autograd.functional.jacobian(
                            lambda x: m.posterior(x).variance.sqrt().T, X
                        )
                        dS[:, i] = (
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
            "mvar_F": np.zeros_like(F),
        }

        return out


class BoTorchSurrogateModelMean(BoTorchSurrogateModel):

    # last objective is the "deterministic" one

    def initialize_model(self, train_x, train_y, train_rho=None):
        # define models for objective and constraint
        train_y_mean = -train_y  # negative because botorch assumes maximization
        train_y_var = train_rho + 1e-10

        models = []
        for i in range(self.n_obj - 1):
            model = CustomHeteroskedasticSingleTaskGP(
                train_X=train_x,
                train_Y=train_y_mean[..., i : i + 1],
                train_Yvar=train_y_var[..., i : i + 1],
                input_transform=self.input_transform,
                outcome_transform=Standardize(m=1),
            )

            models.append(model)

        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)

        class LinearMean(gpytorch.means.Mean):
            def __init__(self):
                super(LinearMean, self).__init__()

            def forward(self, x):
                """Your forward method."""
                return x[..., 1]

        model_mean = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y_mean[..., -1:],
            train_Yvar=train_y_var[..., -1:] * 0.0,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-10)
            ),
            input_transform=self.input_transform,
            # outcome_transform=Standardize(m=1),
            mean_module=LinearMean(),
            covar_module=ZeroKernel(),
        )
        model_mean.likelihood.noise_covar.noise = 1e-10

        models.append(model_mean)
        model = ModelListGP(*models)

        if self.state_dict is not None:
            try:
                # replace each input_transform with the one from the new model
                for i, m in enumerate(model.models):
                    self.bo_model.models[i].input_transform = m.input_transform
                self.state_dict = self.bo_model.state_dict()

                model.load_state_dict(self.state_dict, strict=False)
            except Exception as e:
                print(e)
                print("failed to load state dict")

        return mll, model
