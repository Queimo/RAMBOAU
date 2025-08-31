import numpy as np
import sys
sys.path.append('..')
sys.path.append('.')
from .problem import RiskyProblem
import pandas as pd
import pathlib


class Experiment4D(RiskyProblem):

    def __init__(self, sigma=0.5, repeat_eval=3):

        self.sigma = np.nan
        #                       C_NaOH/C_ZnCl, C_ZnCl, Q_AC, Q_Air
        self.bounds = np.array([[0.5, 0.1, 4.0, 1.0], 
                                [3.5, 1.0, 10.0, 2.5]])
        self.dim = 4
        self.num_objectives = 3
        
        all_batches_paths = list(pathlib.Path("./problems/data/MT-KBH-004/").rglob("XRD+synthsis_data_b*.xlsx"))
        # sort based on batch number (can be multi digit)
        all_batches_paths.sort(key=lambda x: int(x.stem.split("_")[-1][1:]))
        df = pd.read_excel(all_batches_paths[-1])
        df = df[["id", "C_ZnCl", "C_NaOH/C_ZnCl", "C_NaOH" ,"Aspect Ratio", "Peak Ratio", "Q_AC", "Q_AIR", "N_ZnO"]]
        self.obj_cols = ["Peak Ratio_mean", "Aspect Ratio_mean", "N_ZnO_mean"]
        self.var_cols = ["C_NaOH/C_ZnCl_mean", "C_ZnCl_mean", "Q_AC_mean", "Q_AIR_mean"]
        
        df_mean = df.select_dtypes(include=["float64", "int64"]).groupby("id").mean()
        df_var = df.select_dtypes(include=["float64", "int64"]).groupby("id").var()
        df_mean_var = pd.merge(
            df_mean,
            df_var,
            left_index=True,
            right_index=True,
            suffixes=("_mean", "_var"),
        )
        self.df_mean_var = df_mean_var
        print(df_mean_var)
        print(all_batches_paths[-1], "\n")
        print(all_batches_paths[-1], "\n")
        print(all_batches_paths[-1], "\n")
        # X1 = C_NaOH/C_ZnCl, X2 = C_ZnCl
        # Y1 = Peak Ratio, Y2 = Aspect Ratio, Y3 = C_ZnCl
        self.X = df_mean_var[["C_NaOH/C_ZnCl_mean", "C_ZnCl_mean", "Q_AC_mean", "Q_AIR_mean"]].values
        
        Q_AC = self.X[:, 2]
        C_Zn = self.X[:, 1] 
        C_OHZn = self.X[:, 0] 
        C_OH = C_OHZn * C_Zn
        C_ZnO = np.min(np.array([C_Zn, 0.5 * C_OH]), axis=0)
        N_ZnO = C_ZnO * Q_AC
        N_ZnO
        df_mean_var["N_ZnO_mean"] = N_ZnO
        
        self.Y = (
            -1
            * df_mean_var[
                ["Peak Ratio_mean", "Aspect Ratio_mean", "N_ZnO_mean"]
            ].values
        )  # we assume minimzation
        self.rho = df_mean_var[
            ["Peak Ratio_var", "Aspect Ratio_var", "N_ZnO_var"]
        ].values

        super().__init__(
            n_var=self.dim,
            n_obj=self.num_objectives,
            n_constr=0,
            xl=self.bounds[0, :],
            xu=self.bounds[1, :],
        )

    def _evaluate_F(self, x):
        return self.Y[: x.shape[0], :]

    def _evaluate_rho(self, x):
        return self.rho[: x.shape[0], :]

    def pareto_front(self, n_pareto_points=1000):

        from mobo.utils import find_pareto_front

        Y_paretos = find_pareto_front(self.Y)
        Y_paretos_l = find_pareto_front(self.Y)
        Y_paretos_h = find_pareto_front(self.Y)

        return [Y_paretos, Y_paretos_l, Y_paretos_h]

    def get_domain(self):
        return self.bounds

    def f(self, X):
        return self.Y

    def get_noise_var(self, X):
        return self.rho


if __name__ == "__main__":

    prob = Experiment4D()
