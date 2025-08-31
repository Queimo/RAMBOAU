
from __future__ import annotations

from typing import Optional

import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.models.utils.gpytorch_modules import (
    MIN_INFERRED_NOISE_LEVEL,
)
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import (
    _GaussianLikelihoodBase,
    GaussianLikelihood,
)
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch.mlls.noise_model_added_loss_term import NoiseModelAddedLossTerm
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from torch import Tensor
from typing import Optional

import torch
from linear_operator.operators import LinearOperator, MatmulLinearOperator, RootLinearOperator
from torch import Tensor
import gpytorch

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

from botorch.models.gp_regression import (
    HeteroskedasticSingleTaskGP,
    SingleTaskGP,
)



class ZeroKernel(gpytorch.kernels.LinearKernel):
    def __init__(self):        
        super(ZeroKernel, self).__init__()
        
    def forward(
        self, x1: Tensor, x2: Tensor, diag: Optional[bool] = False, last_dim_is_batch: Optional[bool] = False, **params
    ) -> LinearOperator:
        x1_ = x1 
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLinearOperator when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLinearOperator(x1_) * 0.0

        else:
            x2_ = x2 
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

            prod = MatmulLinearOperator(x1_, x2_.transpose(-2, -1))*0.0

        if diag:
            return prod.diagonal(dim1=-1, dim2=-2)*0.0
        else:
            return prod*0.0
        
class CustomHeteroskedasticSingleTaskGP2(SingleTaskGP):
    r"""A single-task exact GP model using a heteroskeastic noise model.

    This model internally wraps another GP (a SingleTaskGP) to model the
    observation noise. This allows the likelihood to make out-of-sample
    predictions for the observation noise levels.
    """

    def __init__(
            self,
            train_X: Tensor,
            train_Y: Tensor,
            train_Yvar: Tensor,
            outcome_transform: Optional[OutcomeTransform] = None,
            input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""A single-task exact GP model using a heteroskedastic noise model.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: A `batch_shape x n x m` tensor of observed measurement
                noise.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
                Note that the noise model internally log-transforms the
                variances, which will happen after this transform is applied.
            input_transform: An input transfrom that is applied in the model's
                forward pass.

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
            >>> se = torch.norm(train_X, dim=1, keepdim=True)
            >>> train_Yvar = 0.1 + se * torch.rand_like(train_Y)
            >>> model = HeteroskedasticSingleTaskGP(train_X, train_Y, train_Yvar)
        """
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        validate_input_scaling(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        noise_likelihood = GaussianLikelihood(
            noise_prior=SmoothedBoxPrior(-3, 5, 0.5, transform=torch.log),
            batch_shape=self._aug_batch_shape,
            noise_constraint=GreaterThan(
                MIN_INFERRED_NOISE_LEVEL, transform=None, initial_value=1.0
            ),
        )
        noise_model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Yvar,
            likelihood=noise_likelihood,
            outcome_transform=Log(),
            input_transform=input_transform,
        )

        heteroskedastic_noise = HeteroskedasticNoise(
            noise_model=noise_model,
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, transform=torch.exp, inv_transform=torch.log),
        )

        likelihood = _GaussianLikelihoodBase(heteroskedastic_noise)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=input_transform,
        )
        self.register_added_loss_term("noise_added_loss")
        self.update_added_loss_term(
            "noise_added_loss", NoiseModelAddedLossTerm(noise_model)
        )
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.to(train_X)      

class CustomHeteroskedasticSingleTaskGP(HeteroskedasticSingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        outcome_transform = None,
        input_transform = None,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: A `batch_shape x n x m` tensor of observed measurement
                noise.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
                Note that the noise model internally log-transforms the
                variances, which will happen after this transform is applied.
            input_transform: An input transfrom that is applied in the model's
                forward pass.
        """
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        validate_input_scaling(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        noise_likelihood = GaussianLikelihood(
            noise_prior=SmoothedBoxPrior(-3, 5, 0.5, transform=torch.log),
            batch_shape=self._aug_batch_shape,
            # noise_constraint=Interval(
            #     MIN_INFERRED_NOISE_LEVEL, 5., transform=None, initial_value=1.0
            # ),
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, transform=None, initial_value=1.0),
        )
        # noise_model = SingleTaskGP(
        #     train_X=train_X,
        #     train_Y=train_Yvar,
        #     likelihood=noise_likelihood,
        #     outcome_transform=Log(),
        #     input_transform=input_transform,
        # )
        
        noise_model = SingleTaskGP(
            train_X=train_X,
            train_Y=torch.log(train_Yvar),
            # likelihood=noise_likelihood,
            # outcome_transform=Log(),
            input_transform=input_transform,  # NOTE: potential bug here
        )
        mll = ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)
        fit_gpytorch_model(mll)

        heteroskedastic_noise = HeteroskedasticNoise(
            noise_model=noise_model,
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, transform=torch.exp, inv_transform=None),
        )
        
        likelihood = _GaussianLikelihoodBase(heteroskedastic_noise)
        # likelihood = _GaussianLikelihoodBase(
        #     HeteroskedasticNoise(
        #         noise_model=noise_model,
                
        #         )
        #     )
        # This is hacky -- this class used to inherit from SingleTaskGP, but it
        # shouldn't so this is a quick fix to enable getting rid of that
        # inheritance
        SingleTaskGP.__init__(
            # pyre-fixme[6]: Incompatible parameter type
            self,
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=input_transform,
        )
        # self.register_added_loss_term("noise_added_loss")
        # self.update_added_loss_term(
        #     "noise_added_loss", NoiseModelAddedLossTerm(noise_model)
        # )
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.to(train_X)


class CustomHeteroskedasticSingleTaskGP3(SingleTaskGP):
    r"""A single-task exact GP model using a heteroskeastic noise model.

    This model internally wraps another GP (a SingleTaskGP) to model the
    observation noise. This allows the likelihood to make out-of-sample
    predictions for the observation noise levels.
    """

    def __init__(
            self,
            train_X: Tensor,
            train_Y: Tensor,
            train_Yvar: Tensor,
            outcome_transform: Optional[OutcomeTransform] = None,
            input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""A single-task exact GP model using a heteroskedastic noise model.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: A `batch_shape x n x m` tensor of observed measurement
                noise.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
                Note that the noise model internally log-transforms the
                variances, which will happen after this transform is applied.
            input_transform: An input transfrom that is applied in the model's
                forward pass.

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
            >>> se = torch.norm(train_X, dim=1, keepdim=True)
            >>> train_Yvar = 0.1 + se * torch.rand_like(train_Y)
            >>> model = HeteroskedasticSingleTaskGP(train_X, train_Y, train_Yvar)
        """
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        validate_input_scaling(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        noise_likelihood = GaussianLikelihood(
            noise_prior=SmoothedBoxPrior(-3, 5, 0.5, transform=torch.log),
            batch_shape=self._aug_batch_shape,
            noise_constraint=GreaterThan(
                MIN_INFERRED_NOISE_LEVEL, transform=None, initial_value=1.0
            ),
        )
        noise_model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Yvar,
            # likelihood=noise_likelihood,
            outcome_transform=Log(),
            input_transform=input_transform,
        )
        mll = ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)
        fit_gpytorch_model(mll)

        heteroskedastic_noise = HeteroskedasticNoise(
            noise_model=noise_model,
            # noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, transform=torch.exp, inv_transform=torch.log),
        )

        likelihood = _GaussianLikelihoodBase(heteroskedastic_noise)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=input_transform,
        )
        # self.register_added_loss_term("noise_added_loss")
        # self.update_added_loss_term(
        #     "noise_added_loss", NoiseModelAddedLossTerm(noise_model)
        # )
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.to(train_X)      