# plot comparison of hypervolume indicator over all runs for any algorithms

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from arguments import get_vis_args
from utils import get_problem_dir, get_algo_names, defaultColors

from pathlib import Path


def main():
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)
    algo_names = get_algo_names(args)

    n_algo, n_seed, seed = len(algo_names), args.n_seed, args.seed

    # read result csvs
    # calculate average hypervolume indicator across seeds
    ds = [{} for _ in range(n_algo)]
    data_list = [[] for _ in range(n_algo)]
    avgHV = np.zeros(n_algo)
    avgHV_all = [None for _ in range(n_algo)]
    num_init_samples = None
    batch_size = None
    for i in range(n_algo):
        for j in range(n_seed):
            if n_seed == 1: j = seed
            # csv_path = f'{problem_dir}/{algo_names[i]}/{j}/EvaluatedSamples.csv'
            csv_path = Path(problem_dir) / algo_names[i] / str(j) / "EvaluatedSamples.csv"
            df = pd.read_csv(csv_path)
            data_list[i].append(df)
    
    df_HV_list = [pd.DataFrame(d) for d in ds]

    sampleIds = list(range(1, len(df_HV_list[0]) + 1))
    # assert num_init_samples is not None

    # Initialize a Plotly figure
    fig = go.Figure()


    # Define a color palette
    color_palette = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']

    # Loop through each algorithm and plot its data
    for idx, algo in enumerate(algo_names):
        # Assume data for the current algorithm is correctly loaded into `data`
        seed = 0
        data = data_list[idx][seed]
        
        # Group the data by iterID and calculate the mean for Hypervolume and MVaR Hypervolume indicators
        # grouped_data = data.groupby('iterID').agg({
        #     'Hypervolume_indicator': 'mean',
        #     'MVaR_Hypervolume_indicator': 'mean'
        # }).reset_index()
        
        grouped_data = data
        grouped_data['n_eval'] = np.arange(1, len(grouped_data) + 1)
        grouped_data.set_index('n_eval', inplace=True)
        
        # Select a color for the current algorithm
        color = color_palette[idx % len(color_palette)]
        
        # Add Hypervolume trace for the current algorithm
        fig.add_trace(go.Scatter(
            x=grouped_data['iterID'], 
            y=grouped_data['Hypervolume_indicator'],
            mode='lines+markers',
            name=f'Hypervolume {algo}',
            line=dict(color=color)
        ))
        
        # Add MVaR Hypervolume trace for the current algorithm with a dotted line
        fig.add_trace(go.Scatter(
            x=grouped_data['iterID'], 
            y=grouped_data['MVaR_Hypervolume_indicator'],
            mode='lines+markers', 
            name=f'MVaR Hypervolume {algo}',
            line=dict(color=color, dash='dot')
        ))

        # Update the layout for the combined figure
        fig.update_layout(
            title='Evolution of Hypervolume and MVaR Hypervolume Across Algorithms',
            xaxis_title='Iteration',
            yaxis_title='Indicator Value',
            legend_title='Dataset'
        )

    if args.savefig:
        fig.write_image(f'./result/{args.problem}/{args.subfolder}/{args.problem}_hv.png')
    else:
        # fig.show()
        fig.write_html(f'./result/{args.problem}/{args.subfolder}/{args.problem}_hv.html')


if __name__ == '__main__':
    main()
