from abc import ABC, abstractmethod
import numpy as np

'''
Selection methods for new batch of samples to evaluate on real problem
'''

class Selection(ABC):
    '''
    Base class of selection method
    '''
    def __init__(self, batch_size, ref_point=None, **kwargs):
        self.batch_size = batch_size
        self.ref_point = ref_point

    def fit(self, X, Y):
        '''
        Fit the parameters of selection method from data
        '''

    def set_ref_point(self, ref_point):
        self.ref_point = ref_point

    @abstractmethod
    def select(self, solution, surrogate_model, status, transformation):
        '''
        Select new samples from solution obtained by solver
        Input:
            solution['x']: design variables of solution
            solution['y']: acquisition values of solution
            solution['algo']: solver algorithm, having some relevant information from optimization
            surrogate_model: fitted surrogate model
            status['pset']: current pareto set found
            status['pfront]: current pareto front found
            status['hv']: current hypervolume
            transformation: data normalization for surrogate model fitting
            (some inputs may not be necessary for some selection criterion)
        Output:
            X_next: next batch of samples selected
            info: other informations need to be stored or exported, None if not necessary
        '''

from .utils import calc_hypervolume
class HVI(Selection):
    '''
    Hypervolume Improvement
    '''
    def select(self, solution, surrogate_model, status, transformation):

        pred_pset = solution['x']
        val = surrogate_model.evaluate(pred_pset)
        pred_pfront = val['F']
        pred_pset, pred_pfront = transformation.undo(pred_pset, pred_pfront)

        curr_pfront = status['pfront'].copy()
        idx_choices = np.ma.array(np.arange(len(pred_pset)), mask=False) # mask array for index choices
        next_batch_indices = []

        # greedily select indices that maximize hypervolume contribution
        for _ in range(self.batch_size):
            curr_hv = calc_hypervolume(curr_pfront, self.ref_point)
            max_hv_contrib = 0.
            max_hv_idx = -1
            for idx in idx_choices.compressed():
                # calculate hypervolume contribution
                new_hv = calc_hypervolume(np.vstack([curr_pfront, pred_pfront[idx]]), self.ref_point)
                hv_contrib = new_hv - curr_hv
                if hv_contrib > max_hv_contrib:
                    max_hv_contrib = hv_contrib
                    max_hv_idx = idx
            if max_hv_idx == -1: # if all candidates have no hypervolume contribution, just randomly select one
                max_hv_idx = np.random.choice(idx_choices.compressed())

            idx_choices.mask[max_hv_idx] = True # mask as selected
            curr_pfront = np.vstack([curr_pfront, pred_pfront[max_hv_idx]]) # add to current pareto front
            next_batch_indices.append(max_hv_idx)
        next_batch_indices = np.array(next_batch_indices)

        return pred_pset[next_batch_indices], None


class Uncertainty(Selection):
    '''
    Uncertainty
    '''
    def select(self, solution, surrogate_model, status, transformation):

        X = solution['x']
        val = surrogate_model.evaluate(X, std=True)
        Y_std = val['S']
        X = transformation.undo(x=X)

        uncertainty = np.prod(Y_std, axis=1)
        top_indices = np.argsort(uncertainty)[::-1][:self.batch_size]
        return X[top_indices], None


class Random(Selection):
    '''
    Random selection
    '''
    def select(self, solution, surrogate_model, status, transformation):
        X = solution['x']
        X = transformation.undo(x=X)
        random_indices = np.random.choice(len(X), size=self.batch_size, replace=False)
        return X[random_indices], None

class IdentitySelect(Selection):

    def select(self, solution, surrogate_model, status, transformation):
        X = solution['x']
        X = transformation.undo(x=X)
        
        assert len(X) == self.batch_size, "Identity selection requires batch size equal to number of solutions"
        
        return X, None
    
class ExperimentSelect(Selection):
    def select(self, solution, surrogate_model, status, transformation):
        X = solution['x']
        X = transformation.undo(x=X)
        return X, None