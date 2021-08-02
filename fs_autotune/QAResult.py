#!/usr/bin/env python3
import os
import sys
import logging
#import subprocess as sp
#import pickle
import traceback
#import shutil
#import re

import pandas as pd

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


class QAResultDB():

    """QA Result DB. """

    def __init__(self, dir_path = "__tuning__", job_file_format="csv"):
        self.logger = logging.getLogger(type(self).__name__)
        self.dir_path = dir_path
        
        self.logger.info("init and load QAResultDB")
        self.qa_data = get_or_create_QAData()
        self.result_data = get_or_create_QAResultData()

        self.logger.info("init and load QAResultDB...Done")

        #self.idx_df = pd.DataFrame()
        #self.job_dfs = {}

    def load(self):
        """Load"""
        self.logger.info("load QAResultDB")
        self.qa_data.load()
        self.result_data.load()

    def save(self):
        """save to_csv.

        """
        self.logger.info("save QAResultDB")
        self.qa_data.save()
        self.result_data.save()

    def add_qa_data(self, name, Cmd = None, Ct=None, Cc=None, CD=None, SP=None):
        """add qa job info data.

        """
        self.qa_data.add(name, Cmd, Ct, Cc, CD, SP)
    
    def get_CDSP(self, name):
        """get CD and SP.

        :name: Case name
        :returns: (CD, SP)

        """
        return self.qa_data.get_CDSP(name)

    def get_name_by_CDSP_range(self, CD1, CD2, SP1, SP2):
        """get name by CD SP range

        :CD: TODO
        :SP: TODO
        :returns: name list

        """
        return self.qa_data.get_name_by_CDSP_range(CD1, CD2, SP1, SP2)

    def add_result(self, name, params, Cmd = None, Ct_err=None, Cc_err=None, CD=None, SP=None):
        """add result.

        """
        return self.result_data.add(name, params, Cmd, Ct_err, Cc_err, CD, SP)
        

    def get_result(self, name, params):
        """get result.

        :name: case name
        :params: parameters
        :returns: (Ct_error, Cc_error, CD, SP)

        """
        return self.result_data.get_result(name, params)
    
    def get_results(self, case_names, params):
        """get name by CD SP range

        :CD: TODO
        :SP: TODO
        :returns: name list

        """
        results = []
        for name in case_names:
            results.append(self.get_result(name, params))
        return results

    def get_results_by_CDSP_range(self, case_names, params, CD1, CD2, SP1, SP2):
        """get name by CD SP range

        :CD: TODO
        :SP: TODO
        :returns: name list

        """
        output_names = []
        results = []
        names = self.get_name_by_CDSP_range(CD1, CD2, SP1, SP2)
        for name in names:
            if name in case_names:
                output_names.append(name)
                results.append(self.get_result(name, params))
        return output_names, results

def get_or_create_QAData():
    """load/create QAData.
    :returns: QAData

    """
    logging.info("get or create QAData")
    qa_data = QAData()
    if not qa_data.loaded:
        for d in os.listdir('data'):
            if not os.path.isdir(os.path.join('data',d)):
                continue
            qa_data.add(d)
        qa_data.loaded = True #XXX
    return qa_data

class QAData:

    """QA job info data """

    def __init__(self, dir_path = "__tuning__"):
        self.data = {}
        self.dir_path = dir_path
        self.filename = os.path.join(self.dir_path, "QAData.csv")
        self.loaded = False

        if os.path.isfile(self.filename):
            self.load()
            self.loaded = True

    def load(self):
        """load.

        """
        df = pd.read_csv(self.filename, index_col=0)
        self.data = df.to_dict(orient='index')
        

    def save(self):
        """save to_csv.

        """
        df = pd.DataFrame.from_dict(self.data, orient='index')
        df.to_csv(self.filename)

    def add(self, name, Cmd = None, Ct=None, Cc=None, CD=None, SP=None):
        """add data.

        """
        if name in self.data:
            data_row = self.data[name]
            if Cmd:
                data_row['Cmd'] = Cmd
            if Ct:
                data_row['Ct'] = Ct
            if Cc:
                data_row['Cc'] = Cc
            if CD:
                data_row['CD'] = CD
            if SP:
                data_row['SP'] = SP
        else:
            data_row = {'Cmd': Cmd, 'Ct': Ct, 'Cc': Cc, 'CD': CD, 'SP': SP}

        self.data[name] = data_row
        
    def get_cmd(self, name):
        """get_cmd.

        :name: case name
        :returns: cmd string or None

        """
        if name in self.data:
            return self.data[name]['Cmd']
        else:
            return None
        

    def get_golden(self, name):
        """TODO: Docstring for get_golden.

        :arg1: TODO
        :returns: TODO

        """
        if name in self.data:
            return self.data[name]['Ct'], self.data[name]['Cc']
        else:
            return None


    def get_CDSP(self, name):
        """get CD and SP.

        :name: Case name
        :returns: (CD, SP)

        """
        if name in self.data:
            if self.data[name]['CD'] and self.data[name]['SP'] and \
                pd.notna(self.data[name]['CD']) and pd.notna(self.data[name]['SP']):
                return self.data[name]['CD'], self.data[name]['SP']
        
        return None

    def get_name_by_CDSP_range(self, CD1, CD2, SP1, SP2):
        """get name by CD SP range

        :CD: TODO
        :SP: TODO
        :returns: name list

        """
        names = []
        #rows = self.data[self.data['CD'] >= CD1 & self.data['CD'] < CD2 & self.data['SP'] >= SP1 & self.data['SP'] >= SP2] #XXX
        #return rows[name].values.tolist()
        for name, data in self.data.items():
            if data['CD'] >= CD1 and data['CD'] < CD2 and data['SP'] >= SP1 and data['SP'] < SP2:
                names.append(name)
        return names

def get_or_create_QAResultData():
    """load/create QAResultData.
    :returns: QAResultData

    """
    logging.info("get or create QAResultData")
    result_data = QAResultData("__tuning__", 'QAResultData.csv')
    return result_data


class QAResultData:
    def __init__(self, dir_path, name):
        self.name = name

        #map[(name, params...)] = [..., Ct_err, Cc_err, CD, CP]
        self.data = {}
        self.dir_path = dir_path
        self.filename = os.path.join(self.dir_path, self.name)
        self.loaded = False
        self.format = ""

        if os.path.isfile(self.filename):
            self.load()
            self.loaded = True

    def load(self):
        """load.

        """
        df = pd.read_csv(self.filename, index_col=0)
        #print(df)
        self.data = df.to_dict(orient='index')
        #print(self.data)

    def save(self):
        """save to csv.

        """
        #print(self.data)
        df = pd.DataFrame.from_dict(self.data, orient='index')
        #print(df)
        df.to_csv(self.filename)

    def add(self, name, params, Cmd = None, Ct_err=None, Cc_err=None, CD=None, SP=None):
        """add result.

        """
        key = str((name, params))
        if key in self.data:
            data_row = self.data[key]
            if Cmd:
                data_row['Cmd'] = Cmd
            if Ct_err:
                data_row['Ct_err'] = Ct_err
            if Cc_err:
                data_row['Cc_err'] = Cc_err
            if CD:
                data_row['CD'] = CD
            if SP:
                data_row['SP'] = SP
        else:
            data_row = {'Cmd': Cmd, 'Ct_err': Ct_err, 'Cc_err': Cc_err, 'CD': CD, 'SP': SP}

        self.data[key] = data_row

    def get_result(self, name, params):
        """get result.

        :name: case name
        :params: parameters
        :returns: (Ct_error, Cc_error, CD, SP)

        """
        key = str((name, params))
        if key in self.data:
            Ct_error = self.data[key]['Ct_err']
            Cc_error = self.data[key]['Cc_err']
            CD = self.data[key]['CD']
            SP = self.data[key]['SP']
            return (Ct_error, Cc_error, CD, SP)
        
        return None



