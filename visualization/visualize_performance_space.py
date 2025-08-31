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

def plotly_grid_plotter(figures=[], path="grid_plots.html", ncols=3):

    # Start HTML string with doctype and head including CSS styles
    html_string = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <title>Plotly Plots</title>
    <style>
    /* Add CSS to style the plot containers */
    .grid-container {
    display: grid;
    """\
    + f"grid-template-columns: repeat({ncols}, 1fr);" + \
    """
    grid-gap: 10px; /* space between plots */
    }
    .grid-item {
    margin: 10px;
    }
    </style>
    </head>
    <body>
    <div class="grid-container">
    """

    # Add the HTML for each figure wrapped in div.grid-item
    for fig in figures:
        
        fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        html_string += f'<div class="grid-item">{fig_html}</div>'

    # Close div.grid-container and body/html tags
    html_string += """
    </div>
    </body>
    </html>
    """

    # Write the HTML to a file
    with open(path, 'w') as file:
        file.write(html_string)


def main():
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)
    algo_names = get_algo_names(args)
    seed = args.seed

    n_algo = len(algo_names)
    problem_name = os.path.basename(os.path.dirname(problem_dir))

    # read result csvs
    data_list, paretoEval_list, paretoGP_list, yml_list = [], [], [], []
    mvar_paretoEval_list, mvar_paretoGP_list = [], []
    for algo_name in algo_names:
        csv_folder = f'{problem_dir}/{algo_name}/{seed}/'
        data_list.append(pd.read_csv(csv_folder + 'EvaluatedSamples.csv'))
        paretoEval_list.append(pd.read_csv(csv_folder + 'ParetoFrontEvaluated.csv'))
        with open(csv_folder + 'args.yml') as f:
            yml_list.append(yaml.load(f, Loader=yaml.SafeLoader))
        paretoGP_list.append(pd.read_csv(csv_folder + 'ParetoFrontApproximation.csv'))
    
    true_front_file = os.path.join(problem_dir, "TrueParetoFront0.csv")
    has_true_front = os.path.exists(true_front_file)
    if has_true_front:
        df_truefront = pd.read_csv(true_front_file)
        
    #get all true front files
    tf_paths = pathlib.Path(problem_dir).glob('TrueParetoFront*.csv')
    df_truefront_list = [pd.read_csv(str(tf_path)) for tf_path in tf_paths]


    n_var = len([key for key in data_list[0] if len(key) == 1 and key <= 'Z' and key >= 'A'])
    n_obj = len([key for key in data_list[0] if key.startswith('f')])

    # calculate proper range of plot
    minX = min([min(df_data['f1']) for df_data in data_list])
    maxX = max([max(df_data['f1']) for df_data in data_list])
    minY = min([min(df_data['f2']) for df_data in data_list])
    maxY = max([max(df_data['f2']) for df_data in data_list])
    if has_true_front:
        minX = min(min(df_truefront['f1']), minX)
        maxX = max(max(df_truefront['f1']), maxX)
        minY = min(min(df_truefront['f2']), minY)
        maxY = max(max(df_truefront['f2']), maxY)
    plot_range_x = [minX - (maxX - minX), maxX + 0.05 * (maxX - minX)]
    plot_range_y = [minY - (maxY - minY), maxY + 0.05 * (maxY - minY)]
    if n_obj > 2:
        minZ = min([min(df_data['f3']) for df_data in data_list])
        maxZ = max([max(df_data['f3']) for df_data in data_list])
        if has_true_front:
            minZ = min(min(df_truefront['f3']), minZ)
            maxZ = max(max(df_truefront['f3']), maxZ)
        plot_range_z = [minZ - (maxZ - minZ), maxZ + 0.05 * (maxZ - minZ)]

    # starting the figure
    fig = [go.Figure() for _ in range(n_algo)]

    # label the sample
    def makeLabel(dfRow):
        retStr = 'Input<br>'
        labels = []
        for i in range(n_var):
            labels.append(f'x{i + 1}')
        for i in range(n_obj):
            label_name = f'Uncertainty_f{i + 1}'
            if label_name in dfRow:
                labels.append(label_name)
            label_name = f'Acquisition_f{i + 1}'
            if label_name in dfRow:
                labels.append(label_name)
        return retStr + '<br>'.join([i+':'+str(round(dfRow[i],2)) for i in labels])

    # set hovertext (label)
    for df in data_list + paretoEval_list + paretoGP_list:
        df['hovertext'] = df.apply(makeLabel, axis=1)

    # Holds the min and max traces for each step

    stepTraces = []
    for kk in range(n_algo):
        stepTrace = []

        # Iterating through all the Potential Steps
        for step in list(set(data_list[kk]['iterID'])): 
            # Trimming our DataFrames to the matching iterID
            data_trimmed = data_list[kk][data_list[kk]['iterID'] < step]
            last_eval = step
            # Getting Data of last evaluated points points
            data_lastevaluated = data_list[kk][data_list[kk]['iterID'] == last_eval]
            # Getting Data of proposed points
            data_proposed = data_list[kk][data_list[kk]['iterID'] == last_eval]
            # First set of samples
            firstsamples = data_list[kk][data_list[kk]['iterID'] == 0]
            paretoEval_trimmed = paretoEval_list[kk][paretoEval_list[kk]['iterID'] == step]
            paretoGP_trimmed = paretoGP_list[kk][paretoGP_list[kk]['iterID'] == step]
            traceStart = len(fig[kk].data)

            scatter = go.Scatter if n_obj == 2 else go.Scatter3d

            # Beginning to add our Traces
            trace_dict = dict(
                name = 'Evaluated Points',
                visible=False,
                mode='markers', 
                x=data_trimmed['f1'], 
                y=data_trimmed['f2'], 
                hovertext=data_trimmed['hovertext'],
                hoverinfo="text",
                marker=dict(
                    color='rgba(0, 0, 255, 0.8)',
                    size=3 if n_obj == 2 else 2
                )
            )
            if n_obj > 2: trace_dict['z'] = data_trimmed['f3']
            fig[kk].add_trace(scatter(**trace_dict))
            
            if yml_list[kk]['general']['ref_point'] is not None:       
                #add reference point
                trace_dict = dict(
                    name = 'Reference Point',
                    visible=False,
                    mode='markers', 
                    x=[yml_list[kk]['general']['ref_point'][0]], 
                    y=[yml_list[kk]['general']['ref_point'][1]], 
                    hovertext=['Reference Point'],
                    hoverinfo="text",
                    marker=dict(
                        color='rgba(0, 0, 0, 1)',
                        size=10,
                        symbol='x'
                    )
                )
                if n_obj > 2: trace_dict['z'] = [yml_list[kk]['general']['ref_point'][2]]
                fig[kk].add_trace(scatter(**trace_dict))
            
            # First set of sample points
            trace_dict = dict(
                name = 'First Set of Sample Points',
                visible=False,
                mode='markers', 
                x=firstsamples['f1'], 
                y=firstsamples['f2'], 
                hovertext=firstsamples['hovertext'],
                hoverinfo="text",
                marker=dict(
                    color='rgba(0, 0, 255, 0)',
                    size=3 if n_obj == 2 else 2,
                    symbol='circle',
                    line=dict(
                        color='rgb(10, 50, 10)',
                        width=1
                    )
                )
            )
            if n_obj > 2: trace_dict['z'] = firstsamples['f3']
            fig[kk].add_trace(scatter(**trace_dict))

            # Adding Trace for Points on Pareto Front
            if n_obj == 2:
                fig[kk].add_trace(scatter(
                    name='Pareto Family',
                    visible=False,
                    mode='markers', 
                    x=paretoGP_trimmed['Pareto_f1'], 
                    y=paretoGP_trimmed['Pareto_f2'], 
                    hovertext = paretoGP_trimmed['hovertext'],
                    hoverinfo="text",
                    marker=dict(
                        # color=10*paretoGP_trimmed['ParetoFamily']+1,
                        size=6,
                        symbol='circle',
                        opacity=0.70
                    )
                ))
            else:
                fig[kk].add_trace(scatter(
                    name='Pareto Front Approximation',
                    visible=False, 
                    mode='markers', 
                    x=paretoGP_trimmed['Pareto_f1'], 
                    y=paretoGP_trimmed['Pareto_f2'], 
                    z=paretoGP_trimmed['Pareto_f3'],
                    hovertext = paretoGP_trimmed['hovertext'],
                    hoverinfo = "text",
                    marker=dict(size=6, symbol='circle', opacity=0.70)
                ))
        
    
            # Evaluated Pareto front points
            trace_dict = dict(
                name = 'Pareto Front Evaluated',
                visible=False,
                mode='markers', 
                x=paretoEval_trimmed['Pareto_f1'], 
                y=paretoEval_trimmed['Pareto_f2'], 
                hovertext=paretoEval_trimmed['hovertext'],
                hoverinfo="text",
                marker=dict(
                    color='yellow',
                    symbol = 'square',
                    size=6 if n_obj == 2 else 4,
                    line=dict(
                        color='rgb(0, 0, 0)',
                        width=1
                    )
                )
            )
            if n_obj > 2: trace_dict['z'] = paretoEval_trimmed['Pareto_f3']
            fig[kk].add_trace(scatter(**trace_dict))

            if 'Expected_f1' in data_proposed:
                # Adding proposed points
                trace_dict = dict(
                    name = 'Expected Proposed Points',
                    visible=False,
                    mode='markers', 
                    x=data_proposed['Expected_f1'], 
                    y=data_proposed['Expected_f2'], 
                    hovertext=data_proposed['hovertext'],
                    hoverinfo="text",
                    marker=dict(
                        color='rgba(255, 0, 0, 0.1)',
                        size=8 if n_obj == 2 else 5,
                        line=dict(
                            color='rgb(255, 50, 10)',
                            width=2
                        )
                    )
                )
                if n_obj > 2: trace_dict['z'] = data_proposed['Expected_f3']
                fig[kk].add_trace(scatter(**trace_dict))
                    
            #Adding last evaluated points
            trace_dict = dict(
                name = 'Evaluated Proposed Points',
                visible=False,
                mode='markers', 
                x=data_lastevaluated['f1'], 
                y=data_lastevaluated['f2'], 
                hovertext=data_lastevaluated['hovertext'],
                hoverinfo="text",
                marker=dict(
                    color='rgba(255, 0, 0, 0.8)',
                    size=9 if n_obj == 2 else 6,
                    symbol='circle',
                    line=dict(
                        color='rgb(10, 50, 10)',
                        width=1
                    )
                )
            )
            if n_obj > 2: trace_dict['z'] = data_lastevaluated['f3']
            fig[kk].add_trace(scatter(**trace_dict))

            # Adding lines between evaluated and proposed performance values
            if step > 0 and 'Expected_f1' in data_proposed:
                for con in range(len(list(data_proposed['Expected_f1']))):
                    trace_dict = dict(
                        #name='Connection between predicted and evaluated performance',
                        visible=False,
                        showlegend=False,
                        mode='lines',
                        x=[list(data_proposed['Expected_f1'])[con], list(data_lastevaluated['f1'])[con]], 
                        y=[list(data_proposed['Expected_f2'])[con], list(data_lastevaluated['f2'])[con]],
                        line=dict(
                            color="MediumPurple",
                            width=1,
                            dash="dot",
                        )
                    )
                    if n_obj > 2: trace_dict['z'] = [list(data_proposed['Expected_f3'])[con],list(data_lastevaluated['f3'])[con]]
                    fig[kk].add_trace(scatter(**trace_dict))

            # Adding true Pareto front points
            
            
            for idx_t, df_truefront in enumerate(df_truefront_list):
                if has_true_front:
                    trace_dict = dict(
                        name="True Pareto Front",
                        visible=False,
                        mode="markers",
                        x=df_truefront["f1"],
                        y=df_truefront["f2"],
                        hovertext=f'True Pareto Front {idx_t}',
                        marker=dict(
                            color=defaultColors[5+idx_t],
                            size=3,
                            symbol="circle",
                        ),
                    )
                    if n_obj > 2:
                        trace_dict["z"] = df_truefront["f3"]
                    fig[kk].add_trace(scatter(**trace_dict))

                
            traceEnd = len(fig[kk].data)-1
            stepTrace.append([i for i in range(traceStart,traceEnd+1)])

        stepTraces.append(stepTrace)

        # Make Last trace visible
        for i in stepTrace[-1]:
            fig[kk].data[i].visible = True
            scene_dict = dict(xaxis=dict(range=plot_range_x), yaxis=dict(range=plot_range_y))
            if n_obj > 2:
                scene_dict['zaxis'] = dict(range=plot_range_z)
            fig[kk].update_layout(scene=scene_dict)

        # Create and add slider
        steps = []
        j = 1
        for stepIndexes in stepTrace:
            # Default set everything Invisivible
            iteration = dict(
                method="restyle",
                args=["visible", [False] * len(fig[kk].data)],
                label=str(j-1)
            )
            j = j + 1
            #Toggle Traces in this Step to Visible
            for i in stepIndexes:
                iteration['args'][1][i] = True
            steps.append(iteration)

        sliders = [dict(
            active=int(len(steps))-1,
            currentvalue={"prefix": "Iteration: "},
            pad={"t": 50},
            steps=steps
        )]

        # Adding some Formatting to the Plot
        scene_dict = dict(
            xaxis_title='f1',
            yaxis_title='f2',
            xaxis = dict(range=plot_range_x),
            yaxis = dict(range=plot_range_y),
            # orthographic projection
            aspectmode='cube',
        )
        if n_obj > 2:
            scene_dict['zaxis_title'] = 'f3'
            scene_dict['zaxis'] = dict(range=plot_range_z)

        fig[kk].update_layout(
            sliders=sliders,
            title=f"Performance Space of {problem_name} using {algo_names[kk]}",
            scene = scene_dict,
            autosize = False,
            showlegend=False,
            width = 900,
            height = 750,
        )
        
        # fig[kk].show()
        # fig[kk].write_html(f'{algo_names[kk]}PerformanceSpace.html')


    plotly_grid_plotter(fig, f'./result/{args.problem}/{args.subfolder}/{args.problem}_seed{seed}_performance_space.html', ncols=2 if n_algo > 1 else 1)
    print(f'./result/{args.problem}/{args.subfolder}/{args.problem}_seed{seed}_performance_space.html')

if __name__ == '__main__':
    main()