import ray
import argparse
from time import time, sleep
from main import run_experiment
# from baselines.nsga2 import run_experiment as run_experiment_nsga2
from arguments import extract_args, get_args
from datetime import datetime
import gc

MAX_NUM_PENDING_TASKS = 6


@ray.remote
def worker(problem, algo, seed, datetime_str, args, framework_args):

    
    #change problem, algo, seed
    args.problem = problem
    args.algo = algo
    args.seed = seed
    
    framework_args["datetime_str"] = datetime_str

    start_time = time()
    
    if algo == 'nsga2':
        # run_experiment_nsga2(args, framework_args)
        pass
    else:
        try:
            run_experiment(args, framework_args)
        except Exception as e:
            print(e)
            print(f'problem {problem} algo {algo} seed {seed} failed, time: {time() - start_time:.2f}s')
            return 0, problem, algo, seed
        
    runtime = time() - start_time
    
    return runtime, problem, algo, seed

def main():
    # ray.init(local_mode=True)
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default=["bstdiag"], nargs='+', help='problems to test')
    parser.add_argument('--algo', type=str, default=["raqneirs", "qnehvi"], nargs='+', help='algorithms to test')
    parser.add_argument('--n-seed', type=int, default=3, help='number of different seeds')
    parser.add_argument('--subfolder', type=str, default='default', help='subfolder name for storing results, directly store under result/ as default')
       
    #parse unknown args
    args, _ = parser.parse_known_args()
    
    start_time = time()
    tasks = []
    
    args_task, framework_args = get_args()
    
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    for seed in range(args.n_seed):
        for problem in args.problem:
            for algo in args.algo:

                if len(tasks) > MAX_NUM_PENDING_TASKS:
                    completed_tasks, tasks = ray.wait(tasks, num_returns=1)
                    runtime, ret_problem, ret_algo, ret_seed = ray.get(completed_tasks[0])
                    if runtime != 0:
                        print(f'problem {ret_problem} algo {ret_algo} seed {ret_seed} done, time: {time() - start_time:.2f}s, runtime: {runtime:.2f}s')

                sleep(1)
                task = worker.remote(problem, algo, seed, datetime_str, args_task, framework_args)
                tasks.append(task)
                print(f'problem {problem} algo {algo} seed {seed} started')
                gc.collect()
    
    while len(tasks) > 0:
        completed_tasks, tasks = ray.wait(tasks, num_returns=1)
        runtime, ret_problem, ret_algo, ret_seed = ray.get(completed_tasks[0])
        if runtime != 0:
            print(f'problem {ret_problem} algo {ret_algo} seed {ret_seed} done, time: {time() - start_time:.2f}s, runtime: {runtime:.2f}s')
    

    print('all experiments done, time: %.2fs' % (time() - start_time))

if __name__ == "__main__":
    main()
