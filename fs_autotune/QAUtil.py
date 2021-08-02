#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess as sp
import multiprocessing as mp
#import pickle
import traceback
import shutil
import re


try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


class QAJobInfo:

    """Docstring for QAJobInfo. """

    def __init__(self, pattern, layer, sram, color):
        """TODO: to be defined. """
        self.pattern = args.pattern
        self.layer = args.layer
        self.sram = args.sram
        self.color = args.color

    def to_str(self):
        """To string
        :returns: TODO

        """
        out_string = ""
        return out_string

class QASettings:
    def __init__(self, args = None):
        #self.verbose = False
        self.cpu_num = None
        self.local = False
        self.force = False
        
        self.manual_tuning_param = {}
        self.manual_tune = False
        self.cost_adjust = 1.0
        self.task_name = None

        if args:
            self.set_from_args(args)

    def set_from_args(self, args):
        # adjust cpu numbers if not set
        if args.local:
            self.local = args.local
            if not args.num:
                self.cpu_num = mp.cpu_count()
        else:
            if not args.num:
                self.cpu_num = 100
            else:
                self.cpu_num = args.num
        
        if args.fail or args.force:
            self.force = True
        
        
        # filtering
        self.pattern = args.pattern
        self.layer = args.layer
        self.sram = args.sram
        self.color = args.color

        #
        self.manual_tuning_param = {}
        if args.manual_tune:
            self.manual_tune = True
            params = args.manual_tune.strip().split('|')
            if len(params) < 3:
                logging.error('worng input format for --manual_tune')
                sys.exit(2)
            self.manual_tuning_param['wext'] = int(params[0])
            self.manual_tuning_param['z0ext'] = int(params[1])
            self.manual_tuning_param['z1ext'] = int(params[2])
            self.manual_tuning_param['zloext'] = int(params[3])
            self.manual_tuning_param['zhiext'] = int(params[4])

            if args.post_tuning:
                self.manual_tuning_param['post_tuning'] = ''
        
        #
        if args.tuning_cost_adj:
            if args.tuning_cost_adj > 0:
                self.cost_adjust = args.tuning_cost_adj

        #
        self.nopool = args.nopool

        # special settings
        self.skip_check_name = args.skip_check_name
        self.skip_check_wxf = args.skip_check_wxf
        self.net_pattern = args.net_pattern
        self.overwrite_fsLog = args.overwrite_fsLog
        self.no_plot = args.no_plot
        self.no_plot_show = args.no_plot_show
        if args.task_name and args.task_name !="":
            self.task_name = args.task_name
        
class QAAnalysisSettings:
    def __init__(self, args = None):
        #self.verbose = False
        self.cpu_num = None
        self.local = False
        self.force = False
        
        self.manual_tuning_param = {}
        self.manual_tune = False
        self.cost_adjust = 1.0
        self.CDSP_range = False
        self.task_name = None

        if args:
            self.set_from_args(args)

    def set_from_args(self, args):
        # adjust cpu numbers if not set
        #if args.local:
        #    self.local = args.local
        #    if not args.num:
        #        self.cpu_num = mp.cpu_count()
        #else:
        #    if not args.num:
        #        self.cpu_num = 100
        #    else:
        #        self.cpu_num = args.num
        #
        #if args.fail or args.force:
        #    self.force = True
        
        
        # filtering
        self.pattern = args.pattern
        self.layer = args.layer
        self.sram = args.sram
        self.color = args.color

        #
        self.manual_tuning_param = {}
        if args.manual_tune:
            self.manual_tune = True
            params = args.manual_tune.strip().split('|')
            if len(params) < 3:
                logging.error('worng input format for --manual_tune')
                sys.exit(2)
            self.manual_tuning_param['wext'] = int(params[0])
            self.manual_tuning_param['z0ext'] = int(params[1])
            self.manual_tuning_param['z1ext'] = int(params[2])
            self.manual_tuning_param['zloext'] = int(params[3])
            self.manual_tuning_param['zhiext'] = int(params[4])

            if args.post_tuning:
                self.manual_tuning_param['post_tuning'] = ''
       
        #
        if args.tuning_cost_adj:
            if args.tuning_cost_adj > 0:
                self.cost_adjust = args.tuning_cost_adj

        #
        self.CDSP_range_param = {}
        if args.CDSP:
            self.CDSP_range = True
            params = args.CDSP.strip().split('|')
            if len(params) < 4:
                logging.error('worng input format for --CDSP')
                sys.exit(3)
            self.CDSP_range_param['CD1'] = int(params[0])
            self.CDSP_range_param['CD2'] = int(params[1])
            self.CDSP_range_param['SP1'] = int(params[2])
            self.CDSP_range_param['SP2'] = int(params[3])
            self.CDSP_range_param['wext'] = None #XXX: update in future

        #
        #self.nopool = args.nopool

        # special settings
        self.skip_check_name = args.skip_check_name
        #self.skip_check_wxf = args.skip_check_wxf
        #self.net_pattern = args.net_pattern
        #self.overwrite_fsLog = args.overwrite_fsLog
        self.no_plot = args.no_plot
        self.no_plot_show = args.no_plot_show
        if args.task_name and args.task_name !="":
            self.task_name = args.task_name

class QAFilter:
    def __init__(self, args):
        self.pattern = args.pattern
        self.layer = args.layer
        self.sram = args.sram
        self.color = args.color

        if args.tuning_cc:
            self.has_cc = True
        else:
            self.has_cc = args.has_cc
        if hasattr(args, "fail"):
            self.fail = args.fail
        else:
            self.fail = False

    def filter_by_name(self, name):
        victim_id, lower_id, higher_id, width, spacing, color, sram = QAUtil.name2tup(name)
        if self.pattern:
            if not re.search( self.pattern, name):
                return False

        if self.fail:
            if not os.path.isfile( os.path.join(name, 'err.log') ):
                return False
            if os.path.isfile( os.path.join(name, 'fs.report') ):
                return False

        if self.has_cc:
            case_path = os.path.join("data", name)
            if not QAUtil.check_case_has_cc(case_path):
                return False

        return self.filter(victim_id, color, sram)

    def filter(self, l_id, color, sram):
        if self.layer:
            if l_id not in self.layer:
                return False

        if self.sram:
            if self.sram == 'n': 
                if sram:
                    return False
            elif self.sram == 'o':
                if not sram:
                    return False
            else:
                if not sram:
                    return False
                elif self.sram != sram.lower() and 'srm_'+ str(self.sram) != sram.lower():
                    return False

        if self.color and color:
            if not color or len(color) != len(self.color):
                return False
            for a,b in zip(self.color, color):
                if a != '?' and a.lower() != b.lower():
                    return False

        return True


class QAUtil:
    def __init__(self, settings):
        self.settings = settings
        self.dirs = []
        self.logger = logging.getLogger(type(self).__name__)

    def get_confirmation(self, msg):
        answer = input('{}? (Y/N)'.format(msg))
        if answer.lower() == 'y' or answer.lower() == 'yes':
            return True
        else:
            return False
    
    def check_exe_callable(self):
        P = sp.run('rapid3d', shell=True, stdout=sp.PIPE, stderr=sp.STDOUT, universal_newlines=True)
        lines = P.stdout.split('\n')
        if len(lines) < 10:
            return False
        else:
            return True
    
    def clean(self):
        for d in self.dirs:
            sp.run('rm -rf {}'.format(os.path.join(d, 'fs.report')), shell=True)
            sp.run('rm -rf {}'.format(os.path.join(d, 'fs.log2')), shell=True)
            sp.run('rm -rf {}'.format(os.path.join(d, 'err.log2')), shell=True)

    def overwrite_fslog(self):
        for d in self.dirs:
            src = os.path.join(d, 'fs.log2')
            des = os.path.join(d, 'fs.log')
            shutil.copy( src, des )

    def get_cases(self):
        return self.dirs

    def collectCase(self, qa_filter):
        #-- collect test cases to run or to display report
        self.dirs = []
        for d in os.listdir('data'):
            if not os.path.isdir(os.path.join('data',d)):
                continue
            if self.settings.skip_check_name: #for non-standard naming cases
                self.dirs.append(d)
            elif self._keepCase(qa_filter, d):
                self.dirs.append(d)

        #-- get confirmation
        for d in self.dirs:
            self.logger.info('collected %s' % d)
            #print('collected ', d)

        return self.dirs

    
    def _keepCase(self, qa_filter, name):
        return qa_filter.filter_by_name(name)

    #XXX: will phase out
    def _keepCase_o(self, ns, name):

        victim_id, lower_id, higher_id, width, spacing, color, sram = self.name2tup(name)

        if ns.pattern:
            if not re.search( ns.pattern, name):
                return False

        if ns.layer:
            if victim_id not in ns.layer:
                return False

        if ns.fail:
            if not os.path.isfile( os.path.join(name, 'err.log') ):
                return False
            if os.path.isfile( os.path.join(name, 'fs.report') ):
                return False

        if ns.sram:
            if ns.sram == 'n': 
                if sram:
                    return False
            elif ns.sram == 'o':
                if not sram:
                    return False
            else:
                if not sram:
                    return False
                elif ns.sram != sram.lower() and 'srm_'+ str(ns.sram) != sram.lower():
                    return False

        if ns.color and color:
            if not color or len(color) != len(ns.color):
                return False
            for a,b in zip(ns.color, color):
                if a != '?' and a.lower() != b.lower():
                    return False

        if ns.has_cc:
            case_path = os.path.join("data", name)
            if not self._check_case_has_cc(case_path):
                return False

        return True
  
    @staticmethod 
    def check_case_has_cc(path, fname='golden'):
        lines = []
        gf_path = os.path.join(path, fname)
        with open(gf_path) as fp:
            lines = fp.readlines()

        cc = float(lines[1])
        if cc > 0:
            return True
        else:
            return False

    @staticmethod
    def name2tup(name):
        nts = name.split('_')
        width = float(nts[1])
        spacing = float(nts[2])
        color = nts[3] if len(nts) >= 4 else None
        sram  = nts[4]+'_'+nts[5] if len(nts)>=5 else None

        header = nts[0].replace('M',',M')[1:].split(',')
        victim = header[0]
        victim_id = int(victim.replace('M',''))
        lower_id = None 
        higher_id = None

        if len(header) >= 2:
            other = header[1]
            other_id = int(other.replace('M',''))
            if other_id > victim_id:
                higher_id = other_id
            else:
                lower_id = other_id

        if len(header) >= 3:
            other = header[2]
            other_id = int(other.replace('M',''))
            if other_id > victim_id:
                higher_id = other_id
            else:
                lower_id = other_id
        return (victim_id, lower_id, higher_id, width, spacing, color, sram) 




