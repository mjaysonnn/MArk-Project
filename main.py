# coding: utf-8
import multiprocessing
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
import argparse

from modules import frontend
from modules import prize_request
from modules import aws_manager
from modules import query_processor
from modules import utils
from modules.instance_source import ins_source
from modules.constants import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=SERVING_PORT)
    parser.add_argument('--signal', type=int, default=0)
    parser.add_argument('--need-updater', type=bool, default=False)
    parser.add_argument('--tag', type=int, default=0)
    return parser.parse_args()

def copy_keys(args, keys):
    src_dict = vars(args)
    return {key: src_dict[key] for key in keys if src_dict[key] is not None}

"""
port -> just port as we know

need_updater -> update on-demand price, spot price

signal -> 0 start application, 1 -> initial settings, 2 -> kill
"""
def main():
    args = get_args()
    params = copy_keys(args, ['port', 'need_updater', 'signal', 'tag'])
    if params['need_updater']:
        multiprocessing.Process(target=prize_request.update_prize).start()  ## update price for demand and spot

    if params['signal'] == 0:
        frontend.main(params['port'], params['tag']) #frontend
    elif params['signal'] == 1:
        ins_source.initial_ins('mx', params['tag'])  # starting initial settings
    elif params['signal'] == 2:
        ins_source.kill_all_ins('mx')
    elif params['signal'] == 3:
        ins_source.launch_backup('mx', params['tag'])
    elif params['signal'] == 4:
        ins_source.stop_backup('mx')

if __name__ == '__main__':
    main()