#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess as sp
import multiprocessing as mp
import pickle
import traceback
import itertools
import shutil
import re
import math
import yaml

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pandas as pd
#import tqdm

#from scipy.optimize import minimize
#from sklearn.neighbors import KNeighborsRegressor

from QAUtil import *
from QAResult import *
from QAJobRunner import *


class TaskFile(object):

    """Docstring for TaskFile. """

    def __init__(self):
        """TODO: to be defined. """
        self.run_jobs = []

    def load(self, fname):
        """load file
        """
        f = open(fname, 'r')
        info_raw = yaml.load(f)

        self.run_jobs = info_raw["run_jobs"]
        self.jobs = info_raw["jobs"]

    def save(self, fname=""):
        """save file
        """
        return #not work currently
        info_raw = {}
        info_raw["run_jobs"] = self.run_jobs
        info_raw["jobs"] = self.jobs
        f = open(fname, 'w')
        yaml.dump(info_raw, f)
        f.close()


class Task):

    """Docstring for Task. """

    def __init__(self, runner, tuner):
        """TODO: to be defined. """
        self.runner = runner
        self.tuner = tuner

    def run(self, arg1):
        """TODO: Docstring for run.

        :arg1: TODO
        :returns: TODO

        """
        pass

    def post_process(self, arg1):
        """TODO: Docstring for post_process.

        :arg1: TODO
        :returns: TODO

        """
        pass


class TaskBuilder:

    """Docstring for TaskBuilder. """

    def __init__(self, settings, task_file = None):
        """TODO: to be defined. """
        self.run_card_name = task_file
        self.run_card = TaskFile()
        #self.info_tmp = None

    def build(self, arg1):
        """TODO: Docstring for build.

        :arg1: TODO
        :returns: TODO

        """
        pass
        #if task_file = None:
        # ceate default runner and tuner


    def create_default_runner_tuner(self, arg1):
        """TODO: Docstring for create_default_runner_tuner.

        :arg1: TODO
        :returns: TODO

        """
        pass

    def load(self):
        """TODO: Docstring for load.
        :returns: TODO

        """
        self.run_card.load(self.run_card_name)
        self.run_jobs = self.run_card.run_jobs
        self.jobs = self.run_card.jobs

    def collect_cases(self, arg1):
        """TODO: Docstring for collect_cases.

        :arg1: TODO
        :returns: TODO

        """
        pass

    def create_runner(self, arg1):
        """TODO: Docstring for create_runner.

        :arg1: TODO
        :returns: TODO

        """
        pass

    def create_tuner(self, args, settings, qa_run):
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

    
