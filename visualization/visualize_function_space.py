# plot each iteration of predicted Pareto front, proposed points, evaluated points for two algorithms
import pathlib
import os, sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from arguments import get_vis_args
from utils import get_problem_dir, get_algo_names, defaultColors
import yaml
from pathlib import Path

def get_data_of_step(pareto_approx_df, selected_iteration):
    filtered_data = pareto_approx_df[pareto_approx_df["iterID"] == selected_iteration]
    return filtered_data


def main():
    # read result csvs
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)
    algo_names = get_algo_names(args)
    seed = args.seed

    n_algo = len(algo_names)
    problem_name = os.path.basename(os.path.dirname(problem_dir))

    # read result csvs
    data_list, paretoEval_list, paretoGP_list, yml_list = [], [], [], []
    for algo_name in algo_names:
        # csv_folder = f"{problem_dir}/{algo_name}/{seed}/"
        csv_folder = Path(problem_dir) / algo_name / str(seed)
        data_list.append(pd.read_csv(csv_folder / "EvaluatedSamples.csv"))
        paretoEval_list.append(pd.read_csv(csv_folder / "ParetoFrontEvaluated.csv"))
        with open(csv_folder / "args.yml") as f:
            yml_list.append(yaml.load(f, Loader=yaml.SafeLoader))
        paretoGP_list.append(pd.read_csv(csv_folder / "ParetoFrontApproximation.csv"))

    # true_front_file = os.path.join(problem_dir, "TrueParetoFront0.csv")
    # has_true_front = os.path.exists(true_front_file)
    true_front_file = Path(problem_dir) / "TrueParetoFront0.csv"
    has_true_front = true_front_file.exists()
    if has_true_front:
        df_truefront = pd.read_csv(true_front_file)

    # get all true front files
    tf_paths = pathlib.Path(problem_dir).glob("TrueParetoFront*.csv")
    df_truefront_list = [pd.read_csv(str(tf_path)) for tf_path in tf_paths]

    n_var = len(
        [key for key in data_list[0] if len(key) == 1 and key <= "Z" and key >= "A"]
    )
    n_obj = len([key for key in data_list[0] if key.startswith("f")])

    for kk, algo in enumerate(algo_names):

        # Create one figure for each seed
        fig = go.Figure()

        approx_all_df = pd.read_csv(f"{problem_dir}/{algo}/{seed}/ApproximationAll.csv")
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

        rows = n_obj-1 if "det" in algo else n_obj
        cols = 4

        fig = make_subplots(
            rows=rows, cols=cols, subplot_titles=("F_i", "S_i", "rho_F_i", "mvar_F_i")
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
                        # zmin=min(approx_all_df[f"F_{i}"]),
                        # zmax=max(approx_all_df[f"F_{i}"]),
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
                        # zmin=min(approx_all_df[f"S_{i}"]),
                        # zmax=max(approx_all_df[f"S_{i}"]),
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
                            # zmin=min(approx_all_df[f"rho_F_{i}"]),
                            # zmax=np.percentile(approx_all_df[f"rho_F_{i}"], 90),
                        ),
                        row=i,
                        col=3,
                    )

                # add mvar_F_1 mvar_F_2
                if hasattr(approx_all_i, f"mvar_F_{i}"):
                    fig.add_trace(
                        go.Contour(
                            x=x[0],
                            y=y[:, 0],
                            z=approx_all_i[f"mvar_F_{i}"].values.reshape(
                                (n_grid, n_grid)
                            ),
                            colorscale="Viridis",
                            showscale=False,
                            visible=(iteration == 1),
                        ),
                        row=i,
                        col=4,
                    )

            for i in range(1, cols + 1):
                for j in range(1, rows + 1):
                    # add evaluation pareto front with yellow squares
                    fig.add_trace(
                        go.Scatter(
                            x=paretoEval_trimmed["x1"],
                            y=paretoEval_trimmed["x2"],
                            mode="markers",
                            visible=(iteration == 1),
                            marker=dict(
                                size=9,
                                color="yellow",
                                symbol="square",
                                line=dict(color="black", width=1),
                            ),
                        ),
                        row=j,
                        col=i,
                    )

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
                            for j in range(int(len(fig.data) / (max_iterations - 1)))
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
        # remove grid and background color
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # fig.show()

        # # Show or save the plot
        # plotly_grid_plotter(fig, f'./result/{args.problem}/{args.subfolder}/{args.problem}_seed{seed}_performance_space.html', ncols=2 if n_algo > 1 else 1)
        if args.savefig:
            fig.write_image(f"{problem_dir}/seed{seed}_{algo}_IO_space.png")
        else:
            fig.write_html(f"{problem_dir}/seed{seed}_{algo}_IO_space.html")

        print(f"Saved {problem_dir}/seed{seed}_{algo}_IO_space")


if __name__ == "__main__":
    main()
