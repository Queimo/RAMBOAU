import os

# os.environ["OMP_NUM_THREADS"] = "1"  # speed up
import numpy as np
from problems.common import build_problem
from mobo.algorithms import get_algorithm
from visualization.data_export import DataExport
from utils import save_args
from utils import RefPoint
import torch
import gc
from mobo.utils import calc_hypervolume

import wandb

def run_experiment(args, framework_args):
    # load arguments

    if "datetime_str" not in framework_args.keys():
        import datetime

        framework_args["datetime_str"] = datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )

    merge_args = {**vars(args), **framework_args}

    name = f"{args.problem}_{args.algo}_{args.seed}_{framework_args['datetime_str']}"
    
    run = wandb.init(project="ramboau", config=merge_args, mode="disabled", name=name)

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # build problem, get initial samples
    problem, true_pfront, X_init, Y_init, rho_init = build_problem(
        args.problem,
        args.n_var,
        args.n_obj,
        args.n_init_sample,
        args.n_process,
        **framework_args["problem_args"],
    )

    args.n_var, args.n_obj = problem.n_var, problem.n_obj

    ref_point_handler = RefPoint(
        Y_init=Y_init,
        rho_init=rho_init,
        alpha=framework_args["problem_args"]["alpha"],
        problem=problem,
    )

    args.ref_point = ref_point_handler.get_ref_point(is_botorch=False)

    print(
        f"Hypervolume of true pareto front 1: {calc_hypervolume(true_pfront[0], args.ref_point)}"
    )
    print(
        f"Hypervolume of true pareto front 1: {calc_hypervolume(true_pfront[1], args.ref_point)}"
    )
    print(
        f"Hypervolume of true pareto front 2: {calc_hypervolume(true_pfront[2], args.ref_point)}"
    )
    # initialize optimizer
    optimizer = get_algorithm(args.algo)(
        problem, args.n_iter, ref_point_handler, framework_args
    )

    # save arguments & setup logger
    save_args(args, framework_args)
    print(problem, optimizer, sep="\n")

    # initialize data exporter
    exporter = DataExport(optimizer, X_init, Y_init, rho_init, args)

    # optimization
    solution = optimizer.solve(X_init, Y_init, rho_init)

    # export true Pareto front to csv
    if true_pfront is not None:
        exporter.write_truefront_csv(true_pfront)

    for step in range(args.n_iter):
        # get new design samples and corresponding performance
        X_next, Y_next, rho_next, Y_next_pred_mean, Y_next_pred_std, acq = next(
            solution
        )
        # update & export current status to csv
        exporter.update(
            X_next, Y_next, Y_next_pred_mean, Y_next_pred_std, acq, rho_next
        )

        # print(exporter.get_wandb_data())
        run.log(exporter.get_wandb_data(args), step=step, commit=False)

        # run subprocess for visualization
        gc.collect()

        exporter.write_csvs()
        exporter.save_psmodel()
    run.log({"final_plot": exporter.wand_final_plot()}, step=step, commit=False)
    run.log({"final_plot2": exporter.wand_final_plot2()}, step=step, commit=True)
    # close logger

    # data['export_pareto'] = wandb.Table(dataframe=self.export_pareto)
    # data['export_approx_pareto'] = wandb.Table(dataframe=self.export_approx_pareto)
    # data['export_data'] = wandb.Table(dataframe=self.export_data)

    # run.summary['export_pareto'] = wandb.Table(dataframe=exporter.export_pareto)
    # run.summary['export_approx_pareto'] = wandb.Table(dataframe=exporter.export_approx_pareto)
    # run.summary['export_data'] = wandb.Table(dataframe=exporter.export_data)

    run.finish()


if __name__ == "__main__":
    from arguments import get_args

    args, framework_args = get_args()

    run_experiment(args, framework_args)
