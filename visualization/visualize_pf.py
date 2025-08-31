import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from arguments import get_vis_args
from utils import get_problem_dir, get_algo_names, defaultColorsCycle, defaultColors
import numpy as np
import os
import pathlib


def get_data_of_step(pareto_approx_df, selected_iteration):
    filtered_data = pareto_approx_df[pareto_approx_df['iterID'] == selected_iteration]
    return filtered_data['Pareto_f1'], filtered_data['Pareto_f2']

def main():
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)
    algo_names = get_algo_names(args)
    subfolder = args.subfolder

    n_algo, n_seed = len(algo_names), args.n_seed
    
    #get all true front files
    tf_paths = pathlib.Path(problem_dir).glob('TrueParetoFront*.csv')
    df_truefront_list = [pd.read_csv(str(tf_path)) for tf_path in tf_paths]
    
    
    for j in range(n_seed):
        # Create one figure for each seed
        fig = go.Figure()
        
        # Add True Pareto Front trace
        # fig.add_trace(go.Scatter(x=true_pareto_df['f1'], y=true_pareto_df['f2'], mode='markers',marker_color="black",name='True Pareto Front'))
        for true_pareto_df in df_truefront_list:
            fig.add_trace(go.Scatter(x=true_pareto_df['f1'], y=true_pareto_df['f2'], mode='markers',marker_color="black",name='True Pareto Front', opacity=0.5))

        # Maximum number of iterations to display
        max_iterations = 20

        # Add algorithm traces for each iteration
        for iteration in range(max_iterations):
            for i in range(n_algo):
                pareto_approx_df = pd.read_csv(f'{problem_dir}/{algo_names[i]}/{j}/ParetoFrontApproximation.csv')
                f1, f2 = get_data_of_step(pareto_approx_df, iteration)
                fig.add_trace(go.Scatter(x=f1, y=f2, mode='markers', visible=(iteration==0),
                                         name=f'{algo_names[i]} - Iter {iteration}',
                                         marker_color=defaultColors[i],
                                         opacity=0.5,
                                         ))

        # Create and add slider
        steps = []
        for iteration in range(max_iterations):
            step = dict(
                method="update",
                args=[{"visible": [True] + [iteration == i for i in range(max_iterations) for _ in range(n_algo)]},
                      {"title": f"Slider switched to iteration: {iteration}"}]
            )
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Iteration: "},
            pad={"t": 10},
            steps=steps
        )]

        fig.update_layout(
            title="True vs Approximate Pareto Front",
            xaxis_title="Objective 1 (f1)",
            yaxis_title="Objective 2 (f2)",
            sliders=sliders
        )

        # Show or save the plot
        if args.savefig:
            fig.write_image(f'{problem_dir}/seed{j}_pareto_front.png')
        else:
            fig.write_html(f'{problem_dir}/seed{j}_pareto_front.html')

        print(f'{problem_dir}/seed{j}_pareto_front.html')

if __name__ == '__main__':
    main()
