import os
import pandas as pd
import numpy as np
from mobo.utils import find_pareto_front, calc_hypervolume, calculate_var
from utils import get_result_dir
import wandb
import os, sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from .arguments import get_vis_args
from .utils import get_problem_dir, get_algo_names, defaultColors
import yaml
import pathlib

"""
Export csv files for external visualization.
"""


class DataExport:
    def __init__(self, optimizer, X, Y, rho, args):
        """
        Initialize data exporter from initial data (X, Y).
        """
        self.optimizer = optimizer
        self.problem = optimizer.real_problem
        self.n_var, self.n_obj = self.problem.n_var, self.problem.n_obj
        self.batch_size = self.optimizer.selection.batch_size
        self.iter = 0
        self.transformation = optimizer.transformation

        # saving path related
        self.result_dir = get_result_dir(args)

        n_samples = X.shape[0]

        # compute initial hypervolume
        pfront, pidx = find_pareto_front(Y, return_index=True)
        pset = X[pidx]
        if args.ref_point is None:
            args.ref_point = optimizer.ref_point_handler.get_ref_point()
        hv_value = calc_hypervolume(pfront, ref_point=args.ref_point)

        # compute MVaR HV
        if not hasattr(self.optimizer.solver, "alpha"):
            alpha = 0.9
        else:
            alpha = self.optimizer.solver.alpha
        self.alpha = alpha
        mvar = calculate_var(Y, variance=rho, alpha=alpha)
        mvar_pfront, mvar_pidx = find_pareto_front(mvar, return_index=True)
        mvar_pset = X[mvar_pidx]
        mvar_hv_value = calc_hypervolume(mvar_pfront, ref_point=args.ref_point)

        # init data frame
        column_names = ["iterID"]
        d1 = {"iterID": np.zeros(n_samples, dtype=int)}
        d2 = {"iterID": np.zeros(len(pset), dtype=int)}
        d5 = {"iterID": np.zeros(len(mvar_pset), dtype=int)}

        # design variables
        for i in range(self.n_var):
            var_name = f"x{i + 1}"
            d1[var_name] = X[:, i]
            d2[var_name] = pset[:, i]
            d5[var_name] = mvar_pset[:, i]
            column_names.append(var_name)

        # performance
        for i in range(self.n_obj):
            obj_name = f"f{i + 1}"
            d1[obj_name] = Y[:, i]
            col_name = f"rho_f{i + 1}"
            d1[col_name] = rho[:, i]
            col_name = f"mvar_f{i + 1}"
            d1[col_name] = mvar[:, i]

            obj_name = f"Pareto_f{i + 1}"
            d2[obj_name] = pfront[:, i]
            d5[obj_name] = mvar_pfront[:, i]

        # predicted performance
        for i in range(self.n_obj):
            obj_pred_name = f"Expected_f{i + 1}"
            d1[obj_pred_name] = np.zeros(n_samples)
            obj_pred_name = f"Uncertainty_f{i + 1}"
            d1[obj_pred_name] = np.zeros(n_samples)
            obj_pred_name = f"Acquisition_f{i + 1}"
            d1[obj_pred_name] = np.zeros(n_samples)

        d1["Hypervolume_indicator"] = np.full(n_samples, hv_value)
        d1["MVaR_Hypervolume_indicator"] = np.full(n_samples, mvar_hv_value)

        self.export_data = pd.DataFrame(data=d1)  # export all data
        self.export_pareto = pd.DataFrame(data=d2)  # export pareto data
        column_names.append("ParetoFamily")
        self.export_approx_pareto = pd.DataFrame(
            columns=column_names
        )  # export pareto approximation data
        self.export_approx_all = pd.DataFrame(
            columns=["iterID"]
        )  # export pareto approximation data
        self.export_mvar_pareto = pd.DataFrame(data=d5)  # export pareto data

        self.has_family = (
            hasattr(self.optimizer.selection, "has_family")
            and self.optimizer.selection.has_family
        )

    def update(
        self, X_next, Y_next, Y_next_pred_mean, Y_next_pred_std, acquisition, rho_next
    ):
        """
        For each algorithm iteration adds data for visualization.
        Input:
            X_next: proposed sample values in design space
            Y_next: proposed sample values in performance space
        """
        self.iter += 1

        # evaluate prediction of X_next on surrogate model
        # val = self.optimizer.surrogate_model.evaluate(self.transformation.do(x=X_next), std=True)
        # Y_next_pred_mean = self.transformation.undo(y=val['F'])
        # Y_next_pred_std = val['S']
        # acquisition, _, _ = self.optimizer.acquisition.evaluate(val)

        pset = self.optimizer.status["pset"]
        pfront = self.optimizer.status["pfront"]
        hv_value = self.optimizer.status["hv"]

        mvar = calculate_var(Y_next, variance=rho_next, alpha=self.alpha)
        mvar_pfront = self.optimizer.status["mvar_pfront"]
        mvar_pset = self.optimizer.status["mvar_pset"]
        mvar_hv_value = self.optimizer.status["mvar_hv"]

        # Y_next_pred_mean = self.optimizer.status['Y_next_pred_mean']
        # Y_next_pred_std = self.optimizer.status['Y_next_pred_std']
        # acquisition = self.optimizer.status['acquisition']

        d1 = {
            "iterID": np.full(self.batch_size, self.iter, dtype=int)
        }  # export all data
        d2 = {
            "iterID": np.full(pfront.shape[0], self.iter, dtype=int)
        }  # export pareto data
        d5 = {
            "iterID": np.full(mvar_pfront.shape[0], self.iter, dtype=int)
        }  # export pareto data

        # design variables
        for i in range(self.n_var):
            var_name = f"x{i + 1}"
            d1[var_name] = X_next[:, i]
            d2[var_name] = pset[:, i]
            d5[var_name] = mvar_pset[:, i]

        # performance and predicted performance
        for i in range(self.n_obj):
            col_name = f"f{i + 1}"
            d1[col_name] = Y_next[:, i]
            d2[f"Pareto_{col_name}"] = pfront[:, i]
            d5[f"Pareto_{col_name}"] = mvar_pfront[:, i]
            
            col_name = f"rho_f{i + 1}"
            d1[col_name] = rho_next[:, i]
            col_name = f"mvar_f{i + 1}"
            d1[col_name] = mvar[:, i]


            col_name = f"Expected_f{i + 1}"
            d1[col_name] = Y_next_pred_mean[:, i]
            col_name = f"Uncertainty_f{i + 1}"
            d1[col_name] = Y_next_pred_std[:, i]
            col_name = f"Acquisition_f{i + 1}"
            d1[col_name] = acquisition[:, i]

        d1["Hypervolume_indicator"] = np.full(self.batch_size, hv_value)
        d1["MVaR_Hypervolume_indicator"] = np.full(self.batch_size, mvar_hv_value)

        if self.has_family:
            info = self.optimizer.info
            family_lbls, approx_pset, approx_pfront = (
                info["family_lbls"],
                info["approx_pset"],
                info["approx_pfront"],
            )
            approx_front_samples = approx_pfront.shape[0]

            d3 = {
                "iterID": np.full(approx_front_samples, self.iter, dtype=int)
            }  # export pareto approximation data

            for i in range(self.n_var):
                var_name = f"x{i + 1}"
                d3[var_name] = approx_pset[:, i]

            for i in range(self.n_obj):
                d3[f"Pareto_f{i + 1}"] = approx_pfront[:, i]

            d3["ParetoFamily"] = family_lbls

        else:
            approx_pset = self.optimizer.solver.solution["x"]
            val = self.optimizer.surrogate_model.evaluate(approx_pset)
            approx_pfront = val["F"]
            approx_pset, approx_pfront = self.transformation.undo(approx_pset, approx_pfront)

            # find undominated
            approx_pfront, pidx = find_pareto_front(approx_pfront, return_index=True)
            approx_pset = approx_pset[pidx]
            approx_front_samples = approx_pfront.shape[0]

            d3 = {"iterID": np.full(approx_front_samples, self.iter, dtype=int)}

            for i in range(self.n_var):
                var_name = f"x{i + 1}"
                d3[var_name] = approx_pset[:, i]

            for i in range(self.n_obj):
                d3[f"Pareto_f{i + 1}"] = approx_pfront[:, i]

            d3["ParetoFamily"] = np.zeros(approx_front_samples)

            # create dataframe from val dict. If field in val is 2d array, then create columns for each element in array
            d4 = {}
            n_grid = 25
            x2 = np.linspace(0, 1, n_grid)
            x1 = np.linspace(0, 1, n_grid)
            x1_mesh, x2_mesh = np.meshgrid(x1, x2)
            x_mesh = np.vstack((x1_mesh.flatten(), x2_mesh.flatten())).T
            if pset.shape[1] > 2:
                # other vars constant 0.5
                x_mesh = np.hstack(
                    (x_mesh, np.full((n_grid**2, pset.shape[1] - 2), 0.5))
                )

            val = self.optimizer.surrogate_model.evaluate(x_mesh, std=True, noise=True)
            
            for key in val:
                if val[key] is None:
                    continue
                if val[key].ndim == 1:
                    d4[key] = val[key]
                else:
                    for i in range(val[key].shape[1]):
                        col_name = f"{key}_{i + 1}"
                        d4[col_name] = val[key][:, i]

            # add iteration id and x1, x2 columns
            d4["iterID"] = np.full(n_grid**2, self.iter, dtype=int)

            X_mesh = self.transformation.undo(x_mesh)
            Y_mesh = val["F"]
            rho_mesh = val["rho_F"]
        
            mvar_F_mesh = calculate_var(Y_mesh, variance=rho_mesh, alpha=self.alpha)

            d4["F_1"] = Y_mesh[:, 0]
            d4["F_2"] = Y_mesh[:, 1]

            d4["mvar_F_1"] = mvar_F_mesh[:, 0]
            d4["mvar_F_2"] = mvar_F_mesh[:, 1]

            d4["x1"] = X_mesh[:, 0]
            d4["x2"] = X_mesh[:, 1]

        df4 = pd.DataFrame(data=d4)
        df5 = pd.DataFrame(data=d5)

        df1 = pd.DataFrame(data=d1)
        df2 = pd.DataFrame(data=d2)
        df3 = pd.DataFrame(data=d3)
        # self.export_data = self.export_data.append(df1, ignore_index=True)
        # self.export_pareto = self.export_pareto.append(df2, ignore_index=True)
        # self.export_approx_pareto = self.export_approx_pareto.append(
        #     df3, ignore_index=True
        # )

        self.export_data = pd.concat([self.export_data, df1], ignore_index=True, axis=0)
        self.export_pareto = pd.concat(
            [self.export_pareto, df2], ignore_index=True, axis=0
        )
        self.export_approx_pareto = pd.concat(
            [self.export_approx_pareto, df3], ignore_index=True, axis=0
        )
        self.export_approx_all = pd.concat(
            [self.export_approx_all, df4], ignore_index=True, axis=0
        )
        self.export_mvar_pareto = pd.concat(
            [self.export_mvar_pareto, df5], ignore_index=True, axis=0
        )

    def save_psmodel(self):
        """
        Save the Pareto set model.
        """
        if not hasattr(self.optimizer.solver, "save_psmodel"):
            return
        self.optimizer.solver.save_psmodel(self.result_dir)

    def write_csvs(self):
        """
        Export data to csv files.
        """
        dataframes = [
            self.export_data,
            self.export_pareto,
            self.export_mvar_pareto,
            self.export_approx_pareto,
            self.export_approx_all,
        ]
        filenames = [
            "EvaluatedSamples",
            "ParetoFrontEvaluated",
            "MVaRParetoFrontEvaluated",
            "ParetoFrontApproximation",
            "ApproximationAll",
        ]

        for dataframe, filename in zip(dataframes, filenames):
            filepath = os.path.join(self.result_dir, filename + ".csv")
            dataframe.to_csv(filepath, index=False)
            
        self.optimizer.surrogate_model.save(self.result_dir)
            


    def write_truefront_csv(self, truefront_list):
        """
        Export true pareto front to csv files.
        """
        problem_dir = os.path.join(
            self.result_dir, "..", ".."
        )  # result/problem/subfolder/

        # if truefront_list is not list
        if not isinstance(truefront_list, list):
            truefront_list = [truefront_list]

        for index, truefront in enumerate(truefront_list):
            filepath = os.path.join(problem_dir, f"TrueParetoFront{index}.csv")

            if os.path.exists(filepath):
                continue

            d = {}
            for i in range(truefront.shape[1]):
                col_name = f"f{i + 1}"
                d[col_name] = truefront[:, i]

            export_tf = pd.DataFrame(data=d)
            export_tf.to_csv(filepath, index=False)

    def get_wandb_data(self, args=None):
        """
        Get data for wandb logging.
        """
        data = {}
        data["iter"] = self.iter
        data["hypervolume"] = self.export_data["Hypervolume_indicator"].iloc[-1]
        data["mvar_hypervolume"] = self.export_data["MVaR_Hypervolume_indicator"].iloc[-1]

        return data

    def wand_final_plot(self, args=None):
        """
        Plot data for wandb logging.
        """
        # get argument values and initializations

        n_algo = 1
        problem_dir = os.path.join(
            self.result_dir, "..", ".."
        )  # result/problem/subfolder/

        problem_name = os.path.basename(os.path.dirname(problem_dir))

        # read result csvs
        data_list, paretoEval_list, paretoGP_list = [], [], []
        data_list.append(self.export_data)
        paretoEval_list.append(self.export_pareto)
        ref_point = self.optimizer.ref_point_handler.get_ref_point()
        paretoGP_list.append(self.export_approx_pareto)

        true_front_file = os.path.join(problem_dir, "TrueParetoFront0.csv")
        has_true_front = os.path.exists(true_front_file)
        if has_true_front:
            df_truefront = pd.read_csv(true_front_file)

        # get all true front files
        tf_paths = pathlib.Path(problem_dir).glob("TrueParetoFront*.csv")
        df_truefront_list = [pd.read_csv(str(tf_path)) for tf_path in tf_paths]

        n_var = len(
            [key for key in data_list[0] if len(key) == 1 and key <= "Z" and key >= "A"]
        )
        n_obj = len([key for key in data_list[0] if key.startswith("f")])

        # calculate proper range of plot
        minX = min([min(df_data["f1"]) for df_data in data_list])
        maxX = max([max(df_data["f1"]) for df_data in data_list])
        minY = min([min(df_data["f2"]) for df_data in data_list])
        maxY = max([max(df_data["f2"]) for df_data in data_list])
        if has_true_front:
            minX = min(min(df_truefront["f1"]), minX)
            maxX = max(max(df_truefront["f1"]), maxX)
            minY = min(min(df_truefront["f2"]), minY)
            maxY = max(max(df_truefront["f2"]), maxY)
        plot_range_x = [minX - (maxX - minX), maxX + 0.05 * (maxX - minX)]
        plot_range_y = [minY - (maxY - minY), maxY + 0.05 * (maxY - minY)]
        if n_obj > 2:
            minZ = min([min(df_data["f3"]) for df_data in data_list])
            maxZ = max([max(df_data["f3"]) for df_data in data_list])
            if has_true_front:
                minZ = min(min(df_truefront["f3"]), minZ)
                maxZ = max(max(df_truefront["f3"]), maxZ)
            plot_range_z = [minZ - (maxZ - minZ), maxZ + 0.05 * (maxZ - minZ)]

        # starting the figure
        fig = [go.Figure() for _ in range(n_algo)]

        # label the sample
        def makeLabel(dfRow):
            retStr = "Input<br>"
            labels = []
            for i in range(n_var):
                labels.append(f"x{i + 1}")
            for i in range(n_obj):
                label_name = f"Uncertainty_f{i + 1}"
                if label_name in dfRow:
                    labels.append(label_name)
                label_name = f"Acquisition_f{i + 1}"
                if label_name in dfRow:
                    labels.append(label_name)
            return retStr + "<br>".join(
                [i + ":" + str(round(dfRow[i], 2)) for i in labels]
            )

        # set hovertext (label)
        for df in data_list + paretoEval_list + paretoGP_list:
            df["hovertext"] = df.apply(makeLabel, axis=1)

        # Holds the min and max traces for each step

        stepTraces = []
        for kk in range(n_algo):
            stepTrace = []

            # Iterating through all the Potential Steps
            for step in list(set(data_list[kk]["iterID"])):
                # Trimming our DataFrames to the matching iterID
                data_trimmed = data_list[kk][data_list[kk]["iterID"] < step]
                last_eval = step
                # Getting Data of last evaluated points points
                data_lastevaluated = data_list[kk][data_list[kk]["iterID"] == last_eval]
                # Getting Data of proposed points
                data_proposed = data_list[kk][data_list[kk]["iterID"] == last_eval]
                # First set of samples
                firstsamples = data_list[kk][data_list[kk]["iterID"] == 0]
                paretoEval_trimmed = paretoEval_list[kk][
                    paretoEval_list[kk]["iterID"] == step
                ]
                paretoGP_trimmed = paretoGP_list[kk][
                    paretoGP_list[kk]["iterID"] == step
                ]
                traceStart = len(fig[kk].data)

                scatter = go.Scatter if n_obj == 2 else go.Scatter3d

                # Beginning to add our Traces
                trace_dict = dict(
                    name="Evaluated Points",
                    visible=False,
                    mode="markers",
                    x=data_trimmed["f1"],
                    y=data_trimmed["f2"],
                    hovertext=data_trimmed["hovertext"],
                    hoverinfo="text",
                    marker=dict(
                        color="rgba(0, 0, 255, 0.8)", size=3 if n_obj == 2 else 2
                    ),
                )
                if n_obj > 2:
                    trace_dict["z"] = data_trimmed["f3"]
                fig[kk].add_trace(scatter(**trace_dict))

                # add reference point
                trace_dict = dict(
                    name="Reference Point",
                    visible=False,
                    mode="markers",
                    x=[ref_point[0]],
                    y=[ref_point[1]],
                    hovertext=["Reference Point"],
                    hoverinfo="text",
                    marker=dict(color="rgba(0, 0, 0, 1)", size=10, symbol="x"),
                )
                if n_obj > 2:
                    trace_dict["z"] = [ref_point[2]]
                fig[kk].add_trace(scatter(**trace_dict))

                # First set of sample points
                trace_dict = dict(
                    name="First Set of Sample Points",
                    visible=False,
                    mode="markers",
                    x=firstsamples["f1"],
                    y=firstsamples["f2"],
                    hovertext=firstsamples["hovertext"],
                    hoverinfo="text",
                    marker=dict(
                        color="rgba(0, 0, 255, 0)",
                        size=3 if n_obj == 2 else 2,
                        symbol="circle",
                        line=dict(color="rgb(10, 50, 10)", width=1),
                    ),
                )
                if n_obj > 2:
                    trace_dict["z"] = firstsamples["f3"]
                fig[kk].add_trace(scatter(**trace_dict))

                # Adding Trace for Points on Pareto Front
                if n_obj == 2:
                    fig[kk].add_trace(
                        scatter(
                            name="Pareto Family",
                            visible=False,
                            mode="markers",
                            x=paretoGP_trimmed["Pareto_f1"],
                            y=paretoGP_trimmed["Pareto_f2"],
                            hovertext=paretoGP_trimmed["hovertext"],
                            hoverinfo="text",
                            marker=dict(
                                # color=10*paretoGP_trimmed['ParetoFamily']+1,
                                size=6,
                                symbol="circle",
                                opacity=0.70,
                            ),
                        )
                    )
                else:
                    fig[kk].add_trace(
                        scatter(
                            name="Pareto Front Approximation",
                            visible=False,
                            mode="markers",
                            x=paretoGP_trimmed["Pareto_f1"],
                            y=paretoGP_trimmed["Pareto_f2"],
                            z=paretoGP_trimmed["Pareto_f3"],
                            hovertext=paretoGP_trimmed["hovertext"],
                            hoverinfo="text",
                            marker=dict(size=6, symbol="circle", opacity=0.70),
                        )
                    )

                # Evaluated Pareto front points
                trace_dict = dict(
                    name="Pareto Front Evaluated",
                    visible=False,
                    mode="markers",
                    x=paretoEval_trimmed["Pareto_f1"],
                    y=paretoEval_trimmed["Pareto_f2"],
                    hovertext=paretoEval_trimmed["hovertext"],
                    hoverinfo="text",
                    marker=dict(
                        color="yellow",
                        symbol="square",
                        size=6 if n_obj == 2 else 4,
                        line=dict(color="rgb(0, 0, 0)", width=1),
                    ),
                )
                if n_obj > 2:
                    trace_dict["z"] = paretoEval_trimmed["Pareto_f3"]
                fig[kk].add_trace(scatter(**trace_dict))

                if "Expected_f1" in data_proposed:
                    # Adding proposed points
                    trace_dict = dict(
                        name="Expected Proposed Points",
                        visible=False,
                        mode="markers",
                        x=data_proposed["Expected_f1"],
                        y=data_proposed["Expected_f2"],
                        hovertext=data_proposed["hovertext"],
                        hoverinfo="text",
                        marker=dict(
                            color="rgba(255, 0, 0, 0.1)",
                            size=8 if n_obj == 2 else 5,
                            line=dict(color="rgb(255, 50, 10)", width=2),
                        ),
                    )
                    if n_obj > 2:
                        trace_dict["z"] = data_proposed["Expected_f3"]
                    fig[kk].add_trace(scatter(**trace_dict))

                # Adding last evaluated points
                trace_dict = dict(
                    name="Evaluated Proposed Points",
                    visible=False,
                    mode="markers",
                    x=data_lastevaluated["f1"],
                    y=data_lastevaluated["f2"],
                    hovertext=data_lastevaluated["hovertext"],
                    hoverinfo="text",
                    marker=dict(
                        color="rgba(255, 0, 0, 0.8)",
                        size=9 if n_obj == 2 else 6,
                        symbol="circle",
                        line=dict(color="rgb(10, 50, 10)", width=1),
                    ),
                )
                if n_obj > 2:
                    trace_dict["z"] = data_lastevaluated["f3"]
                fig[kk].add_trace(scatter(**trace_dict))

                # Adding lines between evaluated and proposed performance values
                if step > 0 and "Expected_f1" in data_proposed:
                    for con in range(len(list(data_proposed["Expected_f1"]))):
                        trace_dict = dict(
                            # name='Connection between predicted and evaluated performance',
                            visible=False,
                            showlegend=False,
                            mode="lines",
                            x=[
                                list(data_proposed["Expected_f1"])[con],
                                list(data_lastevaluated["f1"])[con],
                            ],
                            y=[
                                list(data_proposed["Expected_f2"])[con],
                                list(data_lastevaluated["f2"])[con],
                            ],
                            line=dict(
                                color="MediumPurple",
                                width=1,
                                dash="dot",
                            ),
                        )
                        if n_obj > 2:
                            trace_dict["z"] = [
                                list(data_proposed["Expected_f3"])[con],
                                list(data_lastevaluated["f3"])[con],
                            ]
                        fig[kk].add_trace(scatter(**trace_dict))

                # Adding true Pareto front points
                for df_truefront in df_truefront_list:
                    if has_true_front:
                        trace_dict = dict(
                            name="True Pareto Front",
                            visible=False,
                            mode="markers",
                            x=df_truefront["f1"],
                            y=df_truefront["f2"],
                            marker=dict(
                                size=2,
                                symbol="circle",
                            ),
                        )
                        if n_obj > 2:
                            trace_dict["z"] = df_truefront["f3"]
                        fig[kk].add_trace(scatter(**trace_dict))

                traceEnd = len(fig[kk].data) - 1
                stepTrace.append([i for i in range(traceStart, traceEnd + 1)])

            stepTraces.append(stepTrace)

            # Make Last trace visible
            for i in stepTrace[-1]:
                fig[kk].data[i].visible = True
                scene_dict = dict(
                    xaxis=dict(range=plot_range_x), yaxis=dict(range=plot_range_y)
                )
                if n_obj > 2:
                    scene_dict["zaxis"] = dict(range=plot_range_z)
                fig[kk].update_layout(scene=scene_dict)

            # Create and add slider
            steps = []
            j = 1
            for stepIndexes in stepTrace:
                # Default set everything Invisivible
                iteration = dict(
                    method="restyle",
                    args=["visible", [False] * len(fig[kk].data)],
                    label=str(j - 1),
                )
                j = j + 1
                # Toggle Traces in this Step to Visible
                for i in stepIndexes:
                    iteration["args"][1][i] = True
                steps.append(iteration)

            sliders = [
                dict(
                    active=int(len(steps)) - 1,
                    currentvalue={"prefix": "Iteration: "},
                    pad={"t": 50},
                    steps=steps,
                )
            ]

            # Adding some Formatting to the Plot
            scene_dict = dict(
                xaxis_title="f1",
                yaxis_title="f2",
                xaxis=dict(range=plot_range_x),
                yaxis=dict(range=plot_range_y),
            )
            if n_obj > 2:
                scene_dict["zaxis_title"] = "f3"
                scene_dict["zaxis"] = dict(range=plot_range_z)

            fig[kk].update_layout(
                sliders=sliders,
                title=f"{problem_name}",
                scene=scene_dict,
                autosize=False,
                showlegend=False,
                width=900,
                height=750,
            )

        return fig[0]

    def wand_final_plot2(self, args=None):
        """
        Plot data for wandb logging.
        """
        # get argument values and initializations

        def get_data_of_step(pareto_approx_df, selected_iteration):
            filtered_data = pareto_approx_df[
                pareto_approx_df["iterID"] == selected_iteration
            ]
            return filtered_data

        n_algo = 1
        problem_dir = os.path.join(
            self.result_dir, "..", ".."
        )  # result/problem/subfolder/

        problem_name = os.path.basename(os.path.dirname(problem_dir))

        # read result csvs
        data_list, paretoEval_list, paretoGP_list = [], [], []
        data_list.append(self.export_data)
        paretoEval_list.append(self.export_pareto)
        ref_point = self.optimizer.ref_point_handler.get_ref_point()
        paretoGP_list.append(self.export_approx_pareto)

        true_front_file = os.path.join(problem_dir, "TrueParetoFront0.csv")
        has_true_front = os.path.exists(true_front_file)
        if has_true_front:
            df_truefront = pd.read_csv(true_front_file)

        # get all true front files
        tf_paths = pathlib.Path(problem_dir).glob("TrueParetoFront*.csv")
        df_truefront_list = [pd.read_csv(str(tf_path)) for tf_path in tf_paths]

        n_var = len(
            [key for key in data_list[0] if len(key) == 1 and key <= "Z" and key >= "A"]
        )
        n_obj = len([key for key in data_list[0] if key.startswith("f")])

        for kk in range(n_algo):

            # Create one figure for each seed
            fig = go.Figure()

            approx_all_df = self.export_approx_all

            # label the sample
            def makeLabel(dfRow):
                retStr = "Data:<br>"
                for col in dfRow.index:  # Iterate over all columns
                    retStr += f"{col}: {round(dfRow[col], 2) if type(dfRow[col]) != str else ''}<br>"
                return retStr

            # Set hovertext (label)
            for df in data_list + paretoEval_list + paretoGP_list + [approx_all_df]:
                df["hovertext"] = df.apply(makeLabel, axis=1)
            # Maximum number of iterations to display

            max_iterations = approx_all_df["iterID"].unique().shape[0] + 1

            # get one iteration to check length of data and take square root
            approx_all_i = get_data_of_step(approx_all_df, 1)
            n_grid = int(np.sqrt(approx_all_i.shape[0]))

            rows = 2
            cols = 3

            fig = make_subplots(
                rows=rows, cols=cols, subplot_titles=("F_i", "S_i", "rho_F_i")
            )

            for iteration in range(1, max_iterations):
                approx_all_i = get_data_of_step(approx_all_df, iteration)
                # Trimming our DataFrames to the matching iterID
                data_trimmed = data_list[kk][data_list[kk]["iterID"] < iteration]
                last_eval = iteration
                # Getting Data of last evaluated points points
                data_lastevaluated = data_list[kk][data_list[kk]["iterID"] == last_eval]
                # Getting Data of proposed points
                data_proposed = data_list[kk][data_list[kk]["iterID"] == last_eval]
                # First set of samples
                firstsamples = data_list[kk][data_list[kk]["iterID"] == 0]
                paretoEval_trimmed = paretoEval_list[kk][
                    paretoEval_list[kk]["iterID"] == iteration
                ]
                paretoGP_trimmed = paretoGP_list[kk][
                    paretoGP_list[kk]["iterID"] == iteration
                ]

                # Data reshaping remains the same
                x = approx_all_i["x1"].values.reshape((n_grid, n_grid))
                y = approx_all_i["x2"].values.reshape((n_grid, n_grid))

                for i in range(1, rows + 1):
                    fig.add_trace(
                        go.Contour(
                            x=x[0],
                            y=y[:, 0],
                            z=approx_all_i[f"F_{i}"].values.reshape((n_grid, n_grid)),
                            colorscale="Viridis",
                            showscale=False,
                            visible=(iteration == 1),
                            zmin=min(approx_all_i[f"F_{i}"]),
                            zmax=max(approx_all_i[f"F_{i}"]),
                        ),
                        row=i,
                        col=1,
                    )

                    # add S_1 S_2

                    fig.add_trace(
                        go.Contour(
                            x=x[0],
                            y=y[:, 0],
                            z=approx_all_i[f"S_{i}"].values.reshape((n_grid, n_grid)),
                            colorscale="Viridis",
                            showscale=False,
                            visible=(iteration == 1),
                            zmin=min(approx_all_i[f"S_{i}"]),
                            zmax=max(approx_all_i[f"S_{i}"]),
                        ),
                        row=i,
                        col=2,
                    )

                    # add rho_1 rho_2
                    if hasattr(approx_all_i, f"rho_F_{i}"):
                        fig.add_trace(
                            go.Contour(
                                x=x[0],
                                y=y[:, 0],
                                z=approx_all_i[f"rho_F_{i}"].values.reshape(
                                    (n_grid, n_grid)
                                ),
                                colorscale="Viridis",
                                showscale=False,
                                visible=(iteration == 1),
                                zmin=min(approx_all_i[f"rho_F_{i}"]),
                                zmax=np.percentile(approx_all_i[f"rho_F_{i}"], 90),
                            ),
                            row=i,
                            col=3,
                        )

                for i in range(1, cols + 1):
                    for j in range(1, rows + 1):
                        fig.add_trace(
                            go.Scatter(
                                x=data_trimmed["x1"],
                                y=data_trimmed["x2"],
                                mode="markers",
                                hovertext=data_trimmed["hovertext"],
                                hoverinfo="text",
                                visible=(iteration == 1),
                                marker=dict(
                                    size=8,
                                    color="blue",
                                    symbol="circle",
                                    line=dict(color="black", width=1),
                                ),
                            ),
                            row=j,
                            col=i,
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=firstsamples["x1"],
                                y=firstsamples["x2"],
                                mode="markers",
                                hovertext=firstsamples["hovertext"],
                                hoverinfo="text",
                                visible=(iteration == 1),
                                marker=dict(
                                    size=8,
                                    color="grey",
                                    symbol="circle",
                                    line=dict(color="black", width=1),
                                ),
                            ),
                            row=j,
                            col=i,
                        )

                        # data proposed
                        fig.add_trace(
                            go.Scatter(
                                x=data_proposed["x1"],
                                y=data_proposed["x2"],
                                mode="markers",
                                hovertext=data_proposed["hovertext"],
                                hoverinfo="text",
                                visible=(iteration == 1),
                                marker=dict(
                                    size=15,
                                    color="red",
                                    symbol="x",
                                    line=dict(color="black", width=1),
                                ),
                            ),
                            row=j,
                            col=i,
                        )

                        # add evaluation pareto front with yellow squares
                        fig.add_trace(
                            go.Scatter(
                                x=paretoEval_trimmed["x1"],
                                y=paretoEval_trimmed["x2"],
                                mode="markers",
                                visible=(iteration == 1),
                                marker=dict(
                                    size=8,
                                    color="yellow",
                                    symbol="square",
                                    line=dict(color="black", width=1),
                                ),
                            ),
                            row=j,
                            col=i,
                        )

                        # paretoGP_trimmed low opacity orange circles

                        fig.add_trace(
                            go.Scatter(
                                x=paretoGP_trimmed["x1"],
                                y=paretoGP_trimmed["x2"],
                                mode="markers",
                                visible=(iteration == 1),
                                marker=dict(
                                    size=4, color="orange", symbol="circle", opacity=0.9
                                ),
                            ),
                            row=j,
                            col=i,
                        )

            # Slider setup (similar to your original setup)
            steps = []
            for iteration in range(1, max_iterations):
                step = dict(
                    method="update",
                    args=[
                        {
                            "visible": [
                                iteration == i
                                for i in range(1, max_iterations)
                                for j in range(
                                    int(len(fig.data) / (max_iterations - 1))
                                )
                            ]
                        },
                        {"title": f"Slider switched to iteration: {iteration}"},
                    ],
                )
                steps.append(step)

            sliders = [
                dict(
                    active=0,
                    currentvalue={"prefix": "Iteration: "},
                    pad={"t": 50},
                    steps=steps,
                )
            ]

            fig.update_layout(sliders=sliders)

            return fig
