#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess as sp
#import multiprocessing as mp
#import pickle
import traceback
import argparse
import shutil
#import re

from QAUtil import *
from QAJobRunner import *
from QATuner import *


try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def setup_logger():
    sp.run('rm -rf {}'.format('QA_autotune.log'), shell=True)

    #log file
    logging.basicConfig(level=logging.DEBUG, filename='QA_autotune.log', format='[%(asctime)s] %(levelname)-8s: %(name)-12s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    print('Log file: QA_autotune.log')
    
    # define a Handler which writes messages to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def parse_args():
    parser = argparse.ArgumentParser(description="rapid3d TSMC qualification tuning tool.")
    parser.add_argument('-n', '--num', type=int, help="cpu number")
    parser.add_argument('-l', '--layer', type=int,  nargs='+', help="layer")
    #parser.add_argument('--name', nargs='+')
    #parser.add_argument('-t','--task_file', nargs='+', help="task file for batch mode")
    parser.add_argument('-f', '--force', action='store_true', help="force run rapid3d to update result")
    parser.add_argument('-y', '--confirm', action='store_true', help='direct answer yes to "Ready to submit?"')
    parser.add_argument('--local', action='store_true', help="use local cpu instead of farm")
    parser.add_argument('-p', '--pattern', help="select cases by name pattern.")
    parser.add_argument('-r', '--sram', help="select cases. o: sram only, n: no sram")
    parser.add_argument('-c', '--color', help="select cases by color. e.g. ABA, BBB")
    parser.add_argument('--has_cc', action='store_true',  help="select cases which have cc golden values.")
    parser.add_argument('--tuning_cc', action='store_true', help="autotuning z for Cc issue")
    parser.add_argument('--tuning_z', action='store_true', help="autotuning z for Ct")
    parser.add_argument('--tuning_w', action='store_true', help="autotuning w for Ct")
    parser.add_argument('--tuning_cost_adj', type=float, help="cost value adjustment for autotuning")
    parser.add_argument('--manual_tune', help="format: --manual_tune='wext|z0ext|z1ext|zloext|zhiext'")
    parser.add_argument('--post_tuning', action='store_true', help="apply manual tuning after applying current rules")
    parser.add_argument('--no_plot', action='store_true', help="don't plot")
    parser.add_argument('--no_plot_show', action='store_true', help="only save plot but not show plot")
    parser.add_argument('--task_name', type=str,  nargs='?', help="task name for storing results")

    parser.add_argument('--fail', action='store_true')
    parser.add_argument('--nopool', action='store_true')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--skip_check_name', action='store_true')
    parser.add_argument('--skip_check_wxf', action='store_true')
    parser.add_argument('-np', '--net_pattern')
    parser.add_argument('--overwrite_fsLog', action='store_true')
    args = parser.parse_args()
    return args


def create_tuner(args, settings, qa_run):
    """TODO: Docstring for create_tuner.

    :args: TODO
    :returns: TODO

    """
    if args.tuning_cc:
        qa_tuner = QACcCtDiffTuner(settings, qa_run)
    elif args.tuning_z:
        qa_tuner = QACtZTuner(settings, qa_run)
    elif args.tuning_w:
        qa_tuner = QAWidthTuner(settings, qa_run)
    else:
        qa_tuner = QASingleRunTuner(settings, qa_run)

    return qa_tuner


def main():
    # 1. setup
    args = parse_args()
    setup_logger()
    
    logging.info('fs_autotune start')
    logging.info('Cmd: %s', ' '.join(sys.argv))

    settings = QASettings(args)

    # 2. prepare jobs for running
    qa_filter = QAFilter(args)
    qa_util = QAUtil(settings)
    
    logging.info('Check rapid3d...')
    if qa_util.check_exe_callable() is False:
        logging.error('Calling rapid3d failed ...')
        sys.exit(2)
    logging.info('Check rapid3d...success')

    logging.info('Collect cases...')
    qa_util.collectCase(qa_filter)

    if args.clean:
        qa_util.clean()
        sys.exit(0)

    logging.info('total {} jobs...'.format( len(qa_util.dirs) ))
    print('total {} jobs...'.format( len(qa_util.dirs) ) )
    print('Run with {} cpus'.format(settings.cpu_num))
    if args.confirm:
        logging.info('Job submit confirmed by args')
    elif qa_util.get_confirmation('Ready to submit?'):
        logging.info('Job submit confirmed by user')
    else:
        logging.info('Ready to submit? N')
        sys.exit(0)

    if not os.path.isdir("__tuning__"):
        os.mkdir("__tuning__")
    
    if not os.path.isdir("work"):
        os.mkdir("work")

    if not os.path.isdir("work_tune"):
        os.mkdir("work_tune")


    # 3. job runner
    logging.info('Set job runner')
    qa_run = QAJobRuner(settings)
    
    # 4. tuner
    logging.info('Set and run tuner')
    qa_tuner = create_tuner(args, settings, qa_run)
    qa_tuner.set_cases(qa_util.get_cases())
    qa_tuner.run()

    # 5. post process
    if args.overwrite_fsLog:
        qa_util.overwrite_fsLog()

    logging.info('fs_autotune end')


if __name__ == '__main__':
    main()





