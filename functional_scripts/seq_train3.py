"""
    File to run sequential training simulation

    ### Example of how to run
    GMPS_PATH=/home/gberseth/playground/GMPS MULTIWORLD_PATH=/home/gberseth/playground/multiworld/ python3 functional_scripts/seq_train.py
"""

import sys
import os

GMPS_PATH = os.environ['CoMPS_PATH']
MULTIWORL_PATH = os.environ['MULTIWORLD_PATH']
sys.path.append(GMPS_PATH)
sys.path.append(MULTIWORL_PATH)
from comet_ml import Experiment

comet_logger = Experiment(api_key="KWwx7zh6I2uw6oQMkpEo3smu0",
                           project_name="ml4l3", workspace="glenb")
comet_logger.set_name("test seq train with vpg")

print(comet_logger.get_key())

# comet_logger.end()

import tensorflow as tf
from functional_scripts.remote_train_ppo import experiment as train_experiment
from functional_scripts.local_test_ppo import experiment as rl_experiment

path_to_gmps = GMPS_PATH
test_dir = path_to_gmps + '/zzw_data/'
meta_log_dir = test_dir + '/meta_data/'
EXPERT_DATA_LOC = test_dir + '/seq_expert_traj/'
import numpy as np

def train_seq(meta_variant, rl_variant, comet_logger=comet_logger):
    np.random.seed(0)
    total_tasks = np.arange(0, 40)
    np.random.shuffle(total_tasks)
    comet_exp_key = comet_logger.get_key()
    start_ = 3
    end_ = 10
    # rl_iterations = [2, 4, 6, 8]
    outer_iteration = 0
    for i in range(start_, end_):

        annotation = 'debug-' + str(i) + 'tasks-v0/'

        # policyType = 'conv_fcBiasAda'
        load_policy = None
        # n_meta_itr = meta_variant['n_itr']
        # if (i > start_):
        #     load_policy = meta_log_dir + 'debug-' + str(i - 1) + 'tasks-v0/params.pkl'
        print("TOTAL TASKS:::::::::;", total_tasks)
        meta_variant['log_dir'] = meta_log_dir + annotation
        meta_variant['mbs'] = i
        meta_variant['seed'] = i
        meta_variant['load_policy'] = None
        meta_variant['comet_exp_key'] = comet_exp_key
        meta_variant['outer_iteration'] = outer_iteration
        meta_variant['total_tasks'] = total_tasks

        n_itr = 50
        rl_variant['init_file'] = meta_variant['log_dir'] + '/params.pkl'
        rl_variant['taskIndex'] = total_tasks[i]
        rl_variant['n_itr'] = n_itr

        rl_variant['log_dir'] = EXPERT_DATA_LOC
        rl_variant['outer_iteration'] = outer_iteration
        rl_variant['comet_exp_key'] = comet_exp_key
        outer_iteration += rl_variant['n_itr']
        # outer_iteration += 5


        # train_experiment(variant=meta_variant, comet_exp_key=comet_exp_key)
        # tf.reset_default_graph()
        outer_iteration += meta_variant['n_itr']
        rl_experiment(variant=rl_variant, comet_logger=comet_logger)
        tf.reset_default_graph()
        # outer_iteration += rl_variant['n_itr']



if __name__ == '__main__':
    path_to_gmps = GMPS_PATH
    path_to_multiworld = MULTIWORL_PATH
    # log_dir = path_to_gmps + '/data/Ant_repl/'
    meta_variant = {'policyType': 'fullAda_PPO',
                    'ldim': 4,
                    'init_flr': 0.5,
                    'seed': None,
                    'log_dir': None,
                    'n_parallel': 4,
                    'envType': 'Ant',
                    'fbs': 10,
                    'mbs': None,
                    'max_path_length': 200,
                    'tasksFile': 'rad2_quat_v2',
                    'load_policy': None,
                    'adam_steps': 500,
                    'dagger': None,
                    'expert_policy_loc': None,
                    'use_maesn': False,
                    'expertDataLoc': EXPERT_DATA_LOC,
                    # 'expertDataLoc': path_to_gmps + '/saved_expert_trajs/ant-quat-v2-10tasks-itr400/',
                    'n_itr': 40
                    # 'eval_task_num': 10
                    }

    ############# RL SETTING ############
    expPrefix = 'Test/Ant/'
    policyType = 'PPO'
    if 'conv' in policyType:
        expPrefix = 'img-' + expPrefix

    rl_variant = {'taskIndex': None,
                  'init_file': None,
                  'n_parallel': 4,
                  'log_dir': None,
                  'seed': 1,
                  'tasksFile': 'rad2_quat_v2',
                  'batch_size': 10000,
                  'policyType': policyType,
                  'n_itr': None,
                  'default_step': 0.5,
                  'init_flr': 0.5,
                  'envType': 'Ant',
                  'max_path_length': 200}


    train_seq(meta_variant=meta_variant, rl_variant=rl_variant, comet_logger=comet_logger)
