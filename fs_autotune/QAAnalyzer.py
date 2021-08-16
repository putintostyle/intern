#!/usr/bin/env python3
from itertools import filterfalse
import os
from re import I
import sys
import time
import logging
#import subprocess as sp
#import multiprocessing as mp
#import pickle
import traceback
#import itertools
#import shutil
#import re
import math
from regionselector import *
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pandas as pd
#import tqdm

from scipy.optimize import minimize
from sklearn.neighbors import KNeighborsRegressor

from QAUtil import *
from QAResult import *
from QAJobRunner import *


try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO



class QAModel():

    """Docstring for QAModel. """

    def __init__(self):
        """TODO: to be defined. """
        pass


class QAAnalyzerBase:

    """QAAnalyzer base class. """

    def __init__(self, settings, runner):
        self.settings = settings
        self.job_runner = runner
        self.result_folder = "__tuning__"
        self.qa_db = QAResultDB()
        self.logger = logging.getLogger(type(self).__name__)
        if self.settings.task_name:
            self.result_folder = os.path.join("__tuning__", self.settings.task_name)

    def set_result_folder(self, path="__tuning__"):
        """TODO: Docstring for set_result_folder.

        :path: TODO
        :returns: TODO

        """
        self.result_folder = path

    def set_cases(self, cases):
        """Set job cases.

        :cases: TODO
        :returns: TODO

        """
        self.cases = cases

    def run(self, arg1=None):
        """main for run.

        :arg1: TODO
        :returns: TODO

        """
        raise NotImplementedError("QAAnalyzerBase: run function not implemented!")
    

    def print_statistics(self, results, filename="", brief_stat = False, store = False):
        """main for run.

        :arg1: TODO
        :returns: TODO

        """
        
        self.logger.info('get statistics results')
        if filename and filename != "":
            fh = open(filename, 'w')
        else:
            fh = sys.stdout
        

        result = results

        #--m0 
        Ws, Ss = [], []  
        Ws3, Ss3 = [], []  
        Ws0, Ss0 = [], []  

        diffs = []
        diffAllAbs = 0
        diffAll = 0
        
        ccdiffs = []
        ccdiffAllAbs = 0
        ccdiffAll = 0
        for tcerr, ccerr, aw, asp in result:
            diff = tcerr
            diffAllAbs += abs(diff) ** 2
            diffAll += diff
            diffs.append( diff )
            
            ccdiff = ccerr
            if ccdiff > -100:
                ccdiffAllAbs += abs(ccdiff) ** 2
                ccdiffAll += ccdiff
                ccdiffs.append( ccdiff )

            scaled_diff = abs(diff) * 100
            sign_diff = math.copysign(1, diff)
            if diff < -0.0:
                Ws.append( aw )
                Ss.append( asp )
            else:
                Ws3.append( aw )
                Ss3.append( asp )
            Ws0.append( aw )
            Ss0.append( asp )

        Wall = sorted(Ws + Ws3)
        Sall = sorted(Ss + Ss3)
        diffs = sorted(diffs)
        ccdiffs = sorted(ccdiffs)

        if len(Wall) == 0:
            mean = diffAll
        else:
            mean = diffAll / len(Wall)
        if len(ccdiffs) == 0:
            ccmean = ccdiffAll
        else:
            ccmean = ccdiffAll / len(ccdiffs)

        Wmax = Wall[-1]+100 if len(Wall) > 0 else 1000
        Smax = Sall[-1]+100 if len(Sall) > 0 else 1000

        #print('[Ct]')
        print('   count  n(>=0)  n(<0)     min     max    mean  2-sigma')
        if len(Wall) == 0:
            print('  No Ct data')
        else:
            print( '[Ct] {:3}  {:5}  {:5}   {:6.2f}% {:6.2f}% {:6.2f}% {:6.2f}%'.format(len(result), len(Ws), len(Ws3), diffs[0], diffs[-1], mean, 2 * math.sqrt(diffAllAbs/len(Wall))))
            #print('  total', len(result))
            #print('  diff>=0.0', len(Ws))
            #print('  diff<-0.0', len(Ws3))
            #print('  min {:.2f}%'.format(diffs[0]))
            #print('  max {:.2f}%'.format(diffs[-1]))
            #print('  mean {:.2f}%'.format(mean))
            #print('  2sigma {:.2f}%'.format(2 * math.sqrt(diffAllAbs/len(Wall))))
        
        #print('-'*80)
        #print('[Cc] (exclude -100%)')
        #print('  count                    min     max    mean  2-sigma')
        if len(ccdiffs) == 0:
            print('  No CC data')
        else:
            print( '[Cc] {:3}                 {:6.2f}% {:6.2f}% {:6.2f}% {:6.2f}%'.format(len(ccdiffs), ccdiffs[0], ccdiffs[-1], ccmean, 2 * math.sqrt(ccdiffAllAbs/len(ccdiffs))))
            #print('  total', len(ccdiffs))
            #print('  min {:.2f}%'.format(ccdiffs[0]))
            #print('  max {:.2f}%'.format(ccdiffs[-1]))
            #print('  mean {:.2f}%'.format(ccmean))
            #print('  2sigma {:.2f}%'.format(2 * math.sqrt(ccdiffAllAbs/len(ccdiffs))))
        
    def plot_last_result(self, arg1=None):
        """plot last result.

        :arg1: TODO
        :returns: TODO

        """
        raise NotImplementedError("QAAnalyzerBase: plot_last_result function not implemented!")

    def plot_result(self, result, show_plot=True, save_path="QA_result.png"):
        self.logger.info('Plot results')
        if not result:
            self.logger.warning('No results for plotting!!')
            return

        print()
        print('Plot results') 
        print('-'*80)

        #result = self.last_results

        #--m0 
        Ws, Ss, Cs, As = [], [], [], []  
        Ws3, Ss3, Cs3, As3 = [], [], [], []  
        Ws0, Ss0, Cs0, As0 = [], [], [], []  

        diffs = []
        diffAllAbs = 0
        diffAll = 0
        
        ccdiffs = []
        ccdiffAllAbs = 0
        ccdiffAll = 0

        for tcerr, ccerr, aw, asp in result:
            diff = tcerr
            diffAllAbs += abs(diff) ** 2
            diffAll += diff
            diffs.append( diff )
            
            ccdiff = ccerr
            if ccdiff > -100:
                ccdiffAllAbs += abs(ccdiff) ** 2
                ccdiffAll += ccdiff
                ccdiffs.append( ccdiff )

            scaled_diff = abs(diff) * 100
            sign_diff = math.copysign(1, diff)
            if diff < -0.0:
                Ws.append( aw )
                Ss.append( asp )
                Cs.append( scaled_diff )
                As.append( scaled_diff )
            else:
                Ws3.append( aw )
                Ss3.append( asp )
                Cs3.append( scaled_diff )
                As3.append( scaled_diff )
            Ws0.append( aw )
            Ss0.append( asp )
            Cs0.append( sign_diff )
            As0.append( scaled_diff )

        Wall = sorted(Ws + Ws3)
        Sall = sorted(Ss + Ss3)
        diffs = sorted(diffs)
        ccdiffs = sorted(ccdiffs)

        mean = diffAll / len(Wall)
        if len(ccdiffs) == 0:
            ccmean = ccdiffAll
        else:
            ccmean = ccdiffAll / len(ccdiffs)

        Wmax = Wall[-1]+100 if len(Wall) > 0 else 1000
        Smax = Sall[-1]+100 if len(Sall) > 0 else 1000

        fig, (ax1, ax3) = plt.subplots(1, 2)
        ax1.set_title('< 0.0')
        ax3.set_title('>= +0.0')

        ax1.set_xlim( 0, Wmax )
        ax1.set_ylim( 0, Smax )
        ax3.set_xlim( 0, Wmax )
        ax3.set_ylim( 0, Smax )

        print('[Ct]')
        print('  total', len(result))
        print('  diff>=0.0', len(Ws))
        print('  diff<-0.0', len(Ws3))
        print('  min {:.2f}%'.format(diffs[0]))
        print('  max {:.2f}%'.format(diffs[-1]))
        print('  mean {:.2f}%'.format(mean))
        print('  2sigma {:.2f}%'.format(2 * math.sqrt(diffAllAbs/len(Wall))))
        
        print('-'*80)
        print('[Cc] (exclude -100%)')
        if len(ccdiffs) == 0:
            print('  No CC data')
        else:
            print('  total', len(ccdiffs))
            print('  min {:.2f}%'.format(ccdiffs[0]))
            print('  max {:.2f}%'.format(ccdiffs[-1]))
            print('  mean {:.2f}%'.format(ccmean))
            print('  2sigma {:.2f}%'.format(2 * math.sqrt(ccdiffAllAbs/len(ccdiffs))))


        ax1.scatter(Ws, Ss, s=As, cmap=cm.plasma_r, alpha=0.3)
        ax3.scatter(Ws3, Ss3, s=As3, cmap=cm.plasma_r, alpha=0.3)

        self.logger.info('Save result plot to %s' % (save_path)) 
        print('-'*80)
        print('Save result plot to %s' % (save_path)) 
        plt.savefig(save_path, dpi=200)

        #fig2 = plt.figure(2)
        #fig2, (ax1, ax3) = plt.subplots(1, 2)
        fig2, ax_ = plt.subplots()
        ax_.set_xlim( 0, Wmax )
        ax_.set_ylim( 0, Smax )
        ax_.scatter(Ws0, Ss0, s=As0, c=Cs0, cmap=cm.plasma_r, alpha=0.3)
        plt.savefig(save_path+"_2.png", dpi=200)

        if show_plot:
            plt.show()

    def plot_region(self, result, ref = None, region = None, show = False, select = False):
        # region = [CD1, CD2], [SP1, SP2]
        cal = [list(i) for i in result]
        cal = np.array(cal)
        cal_CD = cal[:,2]
        cal_SP = cal[:,3]
        cal_diffscale = cal[:,0]**2
        if ref != None:
            reference = [list(i) for i in ref]
            reference = np.array(reference)
            ref_CD = reference[:,2]
            ref_SP = reference[:,3]
            ref_diffscale = reference[:,0]**2

            ax1 = plt.subplots(121)
            ax1.scatter(ref_CD, ref_SP, s = ref_diffscale, cmap=cm.plasma_r, alpha=0.3)
            rect1 = ax1.Rectancle((min(region[0]), min(region[1])), abs(region[0][0]-region[0][1]), abs(region[1][0]-region[1][1]), fill = False, edgecolor = 'r', linewidth = 1)
            ax1.add_patch(rect1)
            ax2 = plt.subplots(122)
            ax2.scatter(cal_CD, cal_SP, s = cal_diffscale, cmap=cm.plasma_r, alpha=0.3)
            ax2.add_patch(rect1)
        else:
            fig, ax  = plt.subplots()
            ax.scatter(cal_CD, cal_SP, s = cal_diffscale* 100, c=np.sign(cal[:,1]), cmap=cm.plasma_r, alpha=0.3)
            ax.set_xlim(0,1000)
            ax.set_ylim(0,1000)
            if select:
                return ax, fig
        if show:
            plt.show()
       

class QASingleRunAnalyzer(QAAnalyzerBase):

    """For run job 1 time"""

    def __init__(self, settings, runner):
        super().__init__(settings, runner)

    def run(self, arg1 = None):
        """main for run.

        :arg1: TODO
        :returns: TODO

        """
        print('Analyzer start run')
        self.logger.info('Analyzer... Start')
        if not os.path.isdir(self.result_folder): #XXX: check exist or error out
            os.mkdir(self.result_folder)

        self.job_runner.set_cases(self.cases)
        self.job_runner.set_result_db(self.qa_db)
        params = self.job_runner.tuningParam2tup() #XXX
        print(params) #XXX
        if self.settings.CDSP_range:
            CDSP = self.settings.CDSP_range_param
            cases, result_list = self.job_runner.get_results_by_CDSP(CDSP['CD1'], CDSP['CD2'], CDSP['SP1'], CDSP['SP2'])
        else:
            result_list = self.job_runner.get_results()

        #show all layer statistics
        self.print_statistics(result_list) #print on screen
        self.logger.info('Analyzer... End')

        


class QAWidthAnalyzer(QAAnalyzerBase):

    """QAWidthAnalyzer for tuning line width. """

    def __init__(self, settings, runner):
        super().__init__(settings, runner)
        self.wext_fitter = None
        self._previous_cost = None
        self.itr_data_keys = [] #for accessing first run data
        self.resultsList = [] #only for output
        self.resultsList_with_opt_w = [] #for estimating final results
        self.last_results = None
        self.init = None # result that without cal

        #XXX: need to update
        self.diffs = None
        self.dct_dw_s = None
        self.opt_wext = None
        
        #calibrator
        self.calibrator = None

    
    def run(self, calibration = False, brief_stat = False):
        """main for run.

        :arg1: TODO
        :returns: TODO

        """
        print('Analyzer start run')
        self.logger.info('Analyzer... Start')
        if not os.path.isdir(self.result_folder):
            os.mkdir(self.result_folder)
        self.build_regression_data(None)
        self.calculate_expected_wext(None) #XXX: need refactor
        self.output_wext_result()
        self.opt_wext = self.estimate_optimized_wext(None)
        if calibration:
            if self.calibrator:
                if brief_stat:
                    self.apply_calibration_rule(brief_stat)
                else:
                    self.apply_calibration_rule()
            else:
                print("No calibration rule!!")
        else:
            if self.settings.CDSP_range_param['wext'] != None:
                wext = self.settings.CDSP_range_param['wext']
                print("Applying wext: %s" % wext)
            else:
                wext = self.opt_wext
                print("Applying optimized wext: %s" % wext)
            self.apply_w_rule(wext)
        plot_result = not self.settings.no_plot
        show_plot_result = not self.settings.no_plot_show
        #self.output_wext_fit_result(plot_result, show_plot_result)
        self.logger.info('Analyzer... End')

    def region_run(self):
        # before calibarate, after calibarate
        # for do_diff
        # self.nocal = []
        # self.withcal = []
        print('Before Calibration')
        self.build_regression_data(no_print = True)
        
        self.calculate_expected_wext(None) #XXX: need refactor
        self.output_wext_result()
        self.opt_wext = self.estimate_optimized_wext(None)
        print("Optimized wext: %s" % self.opt_wext)
        
        print('-'*80)
        print('After Calibration')
        self._apply_w_rule(calibrator=self.calibrator)
        self.print_statistics(self.resultsList_with_opt_w)
        
        # self.nocal = self.nocal[0]
        # self.withcal = self.withcal[0]
        # print('   count  n(>=0)  n(<0)     min     max    mean  2-sigma')
        
        # print('Before Calibration')
        # print('-'*20)
        # print( '[Ct] {:3}  {:5}  {:5}   {:6.2f}% {:6.2f}% {:6.2f}% {:6.2f}%'.format(self.nocal[0],self.nocal[1],self.nocal[2],self.nocal[3],self.nocal[4], self.nocal[5], self.nocal[6])) 
        # print('-'*80)
        
        # print('After Calibration')
        
        # print( '[Ct] {:3}  {:5}  {:5}   {:6.2f}% {:6.2f}% {:6.2f}% {:6.2f}%'.format(self.withcal[0],self.withcal[1],self.withcal[2],self.withcal[3],self.withcal[4], self.withcal[5], self.withcal[6]))
        
    def cost(self, result):
        """cost function.

        :result: TODO
        :returns: TODO

        """
        #(c_all, m_all, val_all, cc_all, cm_all, cval_all)
        spec = 2.0*self.settings.cost_adjust
        return (result[2]/spec)**2 # normalize spec to 1

    def build_regression_data(self, arg1 = None, no_print = False):
        """build regression data for wext calculation and fitting

        :arg1: TODO
        :returns: TODO

        """
        self.logger.info('build_regression_data')
        # setup itr 0
        converge = False
        delta = 10
        param = 0
        pre_param = 0
        eta = None
        self._previous_cost = None

        f_log = open("%s/QA_wext_build_regression_data.log" % (self.result_folder), 'w')

        self.job_runner.set_cases(self.cases)
        self.job_runner.set_result_db(self.qa_db)
        self.itr_data_keys = []

        # for itr
        for itr in range(2): #XXX
            # run
            #self.job_runner.set_cases(self.cases)
            self.job_runner.settings.manual_tuning_param['wext'] = param
            params = self.job_runner.tuningParam2tup()
            if no_print == False:
                print('-'*80)
                print("itr %s, wext = %s" % (itr, self.job_runner.settings.manual_tuning_param['wext']))
            f_log.write("itr %s, wext = %s\n" % (itr, self.job_runner.settings.manual_tuning_param['wext']))
            
            cases = []
            if self.settings.CDSP_range:
                CDSP = self.settings.CDSP_range_param
                cases, result_list = self.job_runner.get_results_by_CDSP(CDSP['CD1'], CDSP['CD2'], CDSP['SP1'], CDSP['SP2'])
                #print(CDSP['CD1'], CDSP['CD2'], CDSP['SP1'], CDSP['SP2'])
                #print(len(cases))
                keys = list(zip(cases, [params]*len(cases)))
            else:
                result_list = self.job_runner.get_results()
                keys = list(zip(self.cases, [params]*len(self.cases)))

            #print(keys)
            self.itr_data_keys.append(keys)

            #show all layer statistics
            #print(result_list)
            if no_print == False:
                self.print_statistics(result_list)
            else:
                if itr == 0:
                    self.print_statistics(result_list)
            self.logger.info('Analyzer... End')
            
            if self.init == None:
                if itr == 0:
                    self.init = result_list
            
            # get result
            # check converge

            #cost = self.cost(self.job_runner.get_statistics())
            #print("    normalized cost: %s" % (cost))
            #print("    (count, mean, 2sigma, cccount, ccmean, cc2sigma) = %s" % str(self.job_runner.get_statistics()))
            #print()
            #f_log.write("normalized cost: %s\n" % (cost))
            #f_log.write("(count, mean, 2sigma, cccount, ccmean, cc2sigma) = %s" % str(self.job_runner.get_statistics()))
            #f_log.write('\n')

            cost = 1.0 #dummy value
            # update and rerun if not converge
            if self._previous_cost == None:
                delta = 10
                self._previous_cost = cost
                pre_param = param
                param += delta
                continue
            
            pre_param = param
            self._previous_cost = cost
            param += delta

        print('-'*80)
        f_log.close()

    def calculate_expected_wext(self, arg1):
        """calculate expected wext.

        :arg1: TODO
        :returns: TODO

        """
        self.logger.info('calculate_expected_wext')
        
        self.resultsList = []

        #get keys (name, params)
        #get data [(name, params)]
        self.CDSP = []
        self.exp_wext = []
        for i in range(len(self.itr_data_keys[0])):
            key1 = self.itr_data_keys[0][i]
            key2 = self.itr_data_keys[1][i]
            self.CDSP.append(self.qa_db.get_CDSP(key1[0]))
            wext1 = key1[1][0]
            wext2 = key2[1][0]
            Ct_err1 = self.qa_db.get_result(key1[0], key1[1])[0]
            Ct_err2 = self.qa_db.get_result(key2[0], key2[1])[0]
            
            exp_wext = (Ct_err1*wext2 - Ct_err2*wext1)/(Ct_err1 - Ct_err2)
            self.exp_wext.append(exp_wext)
            #print("XXX: %s -> %s" % (self.qa_db.get_CDSP(key1[0]), exp_wext))
            self.logger.debug("  %s -> %s" % (self.qa_db.get_CDSP(key1[0]), exp_wext))

            #for output
            CD = self.qa_db.get_CDSP(key1[0])[0]
            SP = self.qa_db.get_CDSP(key1[0])[1]
            self.resultsList.append((key1[0], exp_wext, CD, SP))
        self.last_results = self.resultsList

    def estimate_optimized_wext(self, arg1):
        """calculate expected wext.

        :arg1: TODO
        :returns: TODO

        """
        self.logger.info('estimate_optimized_wext')

        #get keys (name, params)
        #get data [(name, params)]
        self.CDSP = []
        self.exp_wext = []
        diffs = []
        dct_dw_s = []
        diff_dct_dw = []
        diff_dw_2 = []
        for i in range(len(self.itr_data_keys[0])):
            key1 = self.itr_data_keys[0][i]
            key2 = self.itr_data_keys[1][i]
            self.CDSP.append(self.qa_db.get_CDSP(key1[0]))
            wext1 = key1[1][0]
            wext2 = key2[1][0]
            Ct_err1 = self.qa_db.get_result(key1[0], key1[1])[0]
            Ct_err2 = self.qa_db.get_result(key2[0], key2[1])[0]
            
            #exp_wext = (Ct_err1*wext2 - Ct_err2*wext1)/(Ct_err1 - Ct_err2)
            #self.exp_wext.append(exp_wext)
            #print("XXX: %s -> %s" % (self.qa_db.get_CDSP(key1[0]), exp_wext))
            #self.logger.debug("  %s -> %s" % (self.qa_db.get_CDSP(key1[0]), exp_wext))
            diff = Ct_err1
            dct_dw = (Ct_err1 - Ct_err2)/(wext1 - wext2)
            #print("%s %s" % (diff, dct_dw))
            diffs.append(diff)
            dct_dw_s.append(dct_dw)
            diff_dct_dw.append(diff * dct_dw)
            diff_dw_2.append(dct_dw**2)
            
            ##for output
            #CD = self.qa_db.get_CDSP(key1[0])[0]
            #SP = self.qa_db.get_CDSP(key1[0])[1]
            #self.resultsList.append((key1[0], exp_wext, CD, SP))

        #Estimate optimized wext
        self.diffs = np.array(diffs)
        self.dct_dw_s = np.array(dct_dw_s)
        diff_dct_dw = np.array(diff_dct_dw)
        diff_dw_2 = np.array(diff_dw_2)
        #print(diff_dct_dw)
        #print(diff_dw_2)
        opt_wext = -1.0*diff_dct_dw.sum()/diff_dw_2.sum()
        opt_wext = int(round(opt_wext))
        print("optimized wext = %s" % (opt_wext))

        #Estimate final result after applying optimized wext
        #new_diff = self.dct_dw_s * opt_wext + self.diffs
        #self.resultsList_with_opt_w = []
        #for i in range(len(self.itr_data_keys[0])):
        #    key1 = self.itr_data_keys[0][i]
        #    CD = self.qa_db.get_CDSP(key1[0])[0]
        #    SP = self.qa_db.get_CDSP(key1[0])[1]
        #    self.resultsList_with_opt_w.append((new_diff[i], -100, CD, SP))

        #print('-'*80)
        #print("Applying optimized wext (prediction)")
        #self.print_statistics(self.resultsList_with_opt_w)
        #self.last_results = self.resultsList_with_opt_w
        return opt_wext

    def _apply_w_rule(self, single_wext=None, calibrator=None):
        self.resultsList_with_opt_w = []
        if not calibrator and single_wext == None:
            raise RuntimeError("_apply_w_rule: calibrator or single_wext not defined!!!")
        for i in range(len(self.itr_data_keys[0])):
            key1 = self.itr_data_keys[0][i]
            CD = self.qa_db.get_CDSP(key1[0])[0]
            SP = self.qa_db.get_CDSP(key1[0])[1]
            if calibrator:
                wext = calibrator.check_rule(CD, SP)
            elif single_wext != None:
                wext = single_wext
            new_diff = self.dct_dw_s[i] * wext + self.diffs[i]
            #print(new_diff, wext)
            self.resultsList_with_opt_w.append((new_diff, -100, CD, SP))
            self.last_results = self.resultsList_with_opt_w

    def apply_calibration_rule(self, brief_stat = filterfalse):
        self._apply_w_rule(calibrator=self.calibrator)
        print('-'*80)
        print("Applying calibration w rule (prediction)")

        if brief_stat:
            self.print_statistics(self.resultsList_with_opt_w)
            print('false')
        else:
            self.print_statistics(self.resultsList_with_opt_w)

    def apply_w_rule(self, opt_wext):
        self._apply_w_rule(opt_wext)
        print('-'*80)
        print("Applying wext (prediction)")
        self.print_statistics(self.resultsList_with_opt_w)
        self.last_results = self.resultsList_with_opt_w


    def plot_last_result(self, show_plot=True, save_path="QA_result.png"):
        self.logger.info('Plot last results')
        self.plot_result(self.last_results, show_plot, save_path) #XXX

    def fit_wext_distribution(self, arg1):
        """fit wext distribution with KNN method.

        :arg1: TODO
        :returns: TODO

        """
        self.logger.info('fit_wext_distribution')
        #X = CDSP = [[cd, sp], ]
        X = self.CDSP
        #y = wext = [...]
        y = self.exp_wext

        knn_fit = KNeighborsRegressor(n_neighbors=8)
        knn_fit.fit(X, y)
        self.wext_fitter = knn_fit


    def output_wext_result(self, filename = "QA_result_wext.result.txt"):

        self.logger.info('output_wext_result') 

        fpath = None
        if filename and filename != "":
            fpath = os.path.join(self.result_folder, filename)
            print('Save expected wext to %s/QA_result_wext.result.txt' % (self.result_folder))
            self.logger.info('Save expected wext to %s/QA_result_wext.result.txt' % (self.result_folder))
        
        resultsList = self.resultsList

        if fpath and fpath != "":
            fh = open(fpath, 'w')
        else:
            fh = sys.stdout

        #--- find mean
        N = len(resultsList)
        tcErrTotal =  sum( t[1] for t in resultsList )
        if N == 0:
            tcmean = tcErrTotal
            return # No output
        else:
            tcmean = tcErrTotal / N

        #--  find stderr
        tcerrsq = 0

        print('-'*100, file = fh)
        print("   Name,    vs,    expected_wext,     CD,     SP", file = fh)
        print('-'*100, file = fh)
        prevName = ''
        for name0, tcerr, aw, asp in resultsList:
            tcerrsq += tcerr ** 2

            ts = name0.split('#')
            name = ts[0] 
            if name != prevName:
                # print('-'*40)
                prevName = name
        
            fmts = ['{:40s}']
            vs = []
            for i in range(1,len(ts)):
                key, value = ts[i].split('_')
                fmts.append('{:4d}')
                vs.append(int(value))
            fmts.extend([ '{:7.2f}', '{:7.1f}', '{:7.1f}'])
            fmtStr = ' '.join(fmts)
            print(fmtStr.format(name, *vs, tcerr, aw, asp), file = fh)

        #-- print final result
        print('-'*100, file = fh)
        print('Expected wext mean/2sigma: {:6.2f} {:6.2f}'.format( tcmean, 2 * math.sqrt( tcerrsq/N ) ), file = fh)
        print('-'*100, file = fh)
    

    def plot(self, init = False, select = False, read = False):
        
        
        if init: # read the stat in initial select
            ax, fig = self.plot_region(self.init, select=True)
        else:
            ax, fig = self.plot_region(self.last_results, select=True) 
        wm = window_motion(fig, ax)
        wm.connect()
        plt.show()
        
        if read:
            sel_region = []
            if init:
                for pts in self.init:
                    if (list(pts)[2] >= wm.region[0][0]) & (list(pts)[2] <= wm.region[1][0]) & (list(pts)[3] >= wm.region[0][1]) & (list(pts)[3] <= wm.region[1][1]):
                        sel_region.append(pts)
            else:
                for pts in self.last_results:
                    if (list(pts)[2] >= wm.region[0][0]) & (list(pts)[2] <= wm.region[1][0]) & (list(pts)[3] >= wm.region[0][1]) & (list(pts)[3] <= wm.region[1][1]):
                        sel_region.append(pts)
            self.print_statistics(sel_region)
        elif select:        
            return wm.region


    def output_wext_fit_result(self, plot_result = True, show_plot = True, plot_path = "QA_result_wext_fit.png"):
        """output wext fit result.

        :plot_result:
        :returns: TODO

        """
        self.logger.info('output_wext_fit_result')
        #
        CDSP = np.array(self.CDSP)
        #xy_min = np.amin(CDSP, axis=0)
        xy_max = np.amax(CDSP, axis=0)
        self.logger.debug("(max CD, max SP): %s", (xy_max))

        x_max = (int(xy_max[0]/100.0)+1) * 100
        y_max = (int(xy_max[1]/100.0)+1) * 100
        x_step = 20
        y_step = 20
        x_step_half = x_step/2
        y_step_half = y_step/2

        #meshgrid
        x = np.arange(0 + x_step_half, x_max + x_step_half, x_step)
        y = np.arange(0 + y_step_half, y_max + y_step_half, y_step)
        xx, yy = np.meshgrid(x, y)
        xy = list(zip(xx.flatten(), yy.flatten()))
        
        #predicted_wext = self.wext_fitter.predict([[cd,sp], ...])
        z = self.wext_fitter.predict(xy)
        z_int = np.rint(z)
        Z = np.array(z_int).reshape(xx.shape)


        #save result
        df = pd.DataFrame(Z)
        df.columns = x
        df.index = y
        df.to_csv('%s/QA_result_wext_fit.result.csv' % (self.result_folder))
        print('Save result to %s/QA_result_wext_fit.result.csv' % (self.result_folder))
        self.logger.info('Save result to %s/QA_result_wext_fit.result.csv' % (self.result_folder))
        
        if plot_result:
            fig, ax = plt.subplots()
            im = ax.imshow(Z, interpolation='none', cmap=cm.RdBu_r,
                           origin='lower', extent=[0, x_max, 0, y_max],
                           vmax=abs(Z).max(), vmin=-abs(Z).max())
            fig.colorbar(im)
            
            png_path = os.path.join(self.result_folder, plot_path)
            plt.savefig(png_path, dpi=200)
            print('Save result plot to %s' % (png_path))
            self.logger.info('Save result plot to %s' % (png_path))

            if show_plot:
                plt.show()


class QAZAnalyzerBase(QAAnalyzerBase):

    """QAZAnalyzer base class for minimizing Ct or Cc error difference by adjust z. """

    def __init__(self, settings, runner):
        super().__init__(settings, runner)

        self.regression_method = ""
        self.cases = []
        self._previous_cost = None

    def run(self, arg1=None):
        """main for run.

        :arg1: TODO
        :returns: TODO

        """
        print('Analyzer start run')
        self.logger.info('Tuning... Start')
        if not os.path.isdir(self.result_folder):
            os.mkdir(self.result_folder)

        #self.example_method_tune_z(None)
        self.simplex_method_tune_z()
        
        self.logger.info('Tuning... End')

    def cost_function(self, result):
        """cost function
        """
        raise NotImplementedError("QAZAnalyzerBase: cost function not implemented!")
            
            
    def example_method_tune_z(self, arg1):
        """TODO: Docstring for methodA.

        :arg1: TODO
        :returns: TODO

        """
        # setup itr 0
        converge = False
        delta = 10
        param = 0
        pre_param = 0
        eta = None

        f_log = open("%s/QA_methodA_regression.log" % (self.result_folder), 'w')

        self.job_runner.set_cases(self.cases)
        self.job_runner.set_result_db(self.qa_db)
        # for itr
        for itr in range(5): #XXX
            if converge:
                break

            # run
            #self.job_runner.set_cases(self.cases)
            self.job_runner.settings.manual_tuning_param['zloext'] = param
            self.job_runner.settings.manual_tuning_param['zhiext'] = param
            
            self.job_runner.run()
            self.job_runner.print_results("%s/QA_result_itr_%s.txt" % (self.result_folder, itr))
            #show all layer statistics
            self.job_runner.print_statistics("%s/QA_result_itr_%s.statistics.txt" % (self.result_folder, itr))
            #self.job_runner.plot_records()

            # get result
            # check converge

            cost = self.cost_tune_cc_ct(self.job_runner.get_statistics())
            print("zhiext = %s" % self.job_runner.settings.manual_tuning_param['zhiext'])
            print("itr %s, normalized cost: %s" % (itr, cost))
            print(self.job_runner.get_statistics())
            f_log.write("zhiext = %s\n" % self.job_runner.settings.manual_tuning_param['zhiext'])
            f_log.write("itr %s, normalized cost: %s\n" % (itr, cost))
            f_log.write(str(self.job_runner.get_statistics()))
            f_log.write('\n')

            if cost > 1:
                converge = False
            else:
                converge = True
                print('XXX1: cost:%s' % cost)
                f_log.write('XXX1: cost:%s\n' % cost)
                break

            # update and rerun if not converge
            if self._previous_cost == None:
                delta = 10
                self._previous_cost = cost
                pre_param = param
                param += delta
                continue
            if eta == None:
                eta = abs(cost/(cost - self._previous_cost)*(param - pre_param)/(cost - self._previous_cost)*(param - pre_param))*0.9 # estimate eta
                print('OOO2: eta: %s' % eta)
                f_log.write('OOO2: eta: %s\n' % eta)
            else:
                eta *= 0.95

            delta = int(-1.0 * eta * (cost - self._previous_cost) /(param - pre_param))
            if abs(delta) < 1 or abs(cost - self._previous_cost) < 0.001:
                print('XXX2: delta:%s, param: %s, delta_cost: %s' % (delta, param, abs(cost - self._previous_cost)))
                f_log.write('XXX2: delta:%s, param: %s, delta_cost: %s\n' % (delta, param, abs(cost - self._previous_cost)))
                break
            pre_param = param
            self._previous_cost = cost
            param += delta

        f_log.close()


    def objective_funciton_for_simplex(self, param):
        """objective function

        :param: tuning parameters
        :returns: cost

        """
        # run
        #self.job_runner.set_cases(self.cases)
        param_ = int(round(param[0]))
        self.job_runner.settings.manual_tuning_param['zloext'] = param_
        self.job_runner.settings.manual_tuning_param['zhiext'] = param_
        
        self.job_runner.run()
        self.job_runner.print_results("%s/QA_result_zhiext_%s.txt" % (self.result_folder, param_))
        #show all layer statistics
        self.job_runner.print_statistics("%s/QA_result_zhiext_%s.statistics.txt" % (self.result_folder, param_))
    
        cost = self.cost_function(self.job_runner.get_statistics())
        print("zhiext = %s" % self.job_runner.settings.manual_tuning_param['zhiext'])
        print("normalized cost: %s" % (cost))
        print("(count, mean, 2sigma, cccount, ccmean, cc2sigma) = %s" % str(self.job_runner.get_statistics()))
        print()

        return cost
        


    def simplex_method_tune_z(self):
        """Tune z with Nelder-Mead Simplex algorithm

        :returns: TODO

        """
        # setup itr 0
        converge = False
        delta = 10
        param = 0
        pre_param = 0
        eta = None

        print('    Write regression log to file: %s/QA_simplex_regression.log' % (self.result_folder))
        self.logger.info('Write regression log to file: %s/QA_simplex_regression.log' % (self.result_folder))
        f_log = open("%s/QA_simplex_regression.log" % (self.result_folder), 'w')

        self.job_runner.set_cases(self.cases)
        self.job_runner.set_result_db(self.qa_db)

        x0 = np.array([0.0])
        #res = minimize(self.objective_funciton_for_simplex, x0, method='nelder-mead',
        #        options={'maxiter':5, 'xatol': 0.9, 'disp': True}) #Not work!!!
        dx0 = np.array([10.0]) # first step
        options = {}
        options['log_file'] = f_log
        best_result = simplex(self.objective_funciton_for_simplex, x0, dx0, options = options)
        print('best result: z=%s, normalized cost=%s' % (best_result[0][0], best_result[1]))
        self.logger.info('best result: z=%s, cost=%s' % (best_result[0][0], best_result[1]))
        
        f_log.close()




class QACcCtDiffAnalyzer(QAZAnalyzerBase):

    """QACcCtDiffAnalyzer for minimizing Ct Cc difference by adjust z. """

    def __init__(self, settings, runner):
        super().__init__(settings, runner)

    def cost_function(self, result):
        """cost function for minimizing ct cc difference

        :result: TODO
        :returns: TODO

        """
        #(c_all, m_all, val_all, cc_all, cm_all, cval_all)
        spec = 0.5*self.settings.cost_adjust
        return ((result[1] - result[4] - 0.0)/spec)**2 # normalize 0.5 to 1


class QACtZAnalyzer(QAZAnalyzerBase):

    """QACtZAnalyzer for minimizing Ct error by adjust z. """

    def __init__(self, settings, runner):
        super().__init__(settings, runner)

    def cost_function(self, result):
        """cost function for minimizing ct error

        :result: TODO
        :returns: TODO

        """
        #(c_all, m_all, val_all, cc_all, cm_all, cval_all)
        spec = 2.0*self.settings.cost_adjust
        print("cost_adjust: ", self.settings.cost_adjust)
        print("tuning spec:", spec)
        return (result[2]/spec)**2 # normalize spec 2.0 to 1


def simplex(func, X0, dX0, args=(), options=None):
    """Nelder-Mead Simplex algorithm

    :func: objective function
    :X0: initial point
    :dX0: first step size
    :args: args for objective function
    :options: for optimization contol
    :returns: X_best, Y_best

    """
    f_log = options['log_file']
    X = np.copy(X0)
    n = X.shape[0] #dimenssion of space
    params = (1.0, 2.0, 0.5, 0.5) #alpha, beta, garmma, delta
    if n > 1:
        params = (1.0, 1.0 + 2.0/n, 0.75 - 0.5/n, 1.0 - 1.0/n) #see paper: DOI: 10.1007/s10589-010-9329-3

    f_log.write('n: %s\n' % n)
    f_log.write('params: %s\n' % str(params))

    #create n+1 test points Xs
    if dX0 != None:
        dX = np.copy(dX0)
    else:
        dX = np.ones(n)
    f_log.write('dX: %s\n' % dX)

    dXs = np.zeros((n+1, n))
    for i in range(0, n):
        dx_i = np.zeros(n)
        dx_i[i] = dX[i]
        dXs[i+1] = dx_i[i]
    Xs = np.tile(X, (n+1,1))
    Xs = Xs + dXs
    Ys = None
    f_log.write('Xs: %s\n' % Xs)

    # best result
    X_best = None
    Y_best = None

    # stop criteria
    x_tol = [1.0]*n
    y_tol = 0.1
    y_abs = 0.1
    max_itr = 7 #XXX
    f_log.write('x_tol: %s, y_tol: %s, y_abs: %s, max_itr: %s\n' % (x_tol, y_tol, y_abs, max_itr))
    for itr in range(max_itr):
        print('-'*80)
        print("itr %s" % itr)
        f_log.write("\nitr %s\n" % itr)

        #get cost
        if Ys is None:
            Ys = []
            for i in range(n+1):
                Ys.append(func(Xs[i], *args))
        Ys = np.array(Ys)
        
        #sorting
        sorted_idx = np.argsort(Ys)
        Xs = Xs[sorted_idx]
        Ys = Ys[sorted_idx]

        f_log.write('  Xs: %s\n' % Xs)
        f_log.write('  Ys: %s\n' % Ys)

        #check converge
        if simplex_check_converge(Xs, Ys, x_tol, y_tol, y_abs):
            f_log.write('  Converge\n')
            break

        #calculate Xc
        Xc = Xs[:-1]
        Xc = Xc.mean(axis = 0)

        #calculate Xr
        Xr = Xc - params[0] * (Xs[-1] - Xc)
        Yr = func(Xr, *args)
        f_log.write('  Xr: %s\n' % Xr)
        f_log.write('  Yr: %s\n' % Yr)

        #Reflection
        if Yr >= Ys[0] and Yr < Ys[-2]:
            Xs[-1] = Xr
            Ys[-1] = Yr
            continue
        
        #Expansion
        if Yr < Ys[0]:
            #Xe
            Xe = Xc + params[1] * (Xr - Xc)
            Ye = func(Xe, *args)
            f_log.write('  Xe: %s\n' % Xe)
            f_log.write('  Ye: %s\n' % Ye)
            if Ye < Yr:
                Xs[-1] = Xe
                Ys[-1] = Ye
            else:
                Xs[-1] = Xr
                Ys[-1] = Yr
            continue

        #Outside Contraction
        if Yr >= Ys[-2] and Yr < Ys[-1]:
            Xoc = Xc + params[2] * (Xr - Xc)
            Yoc = func(Xoc, *args)
            f_log.write('  Xoc: %s\n' % Xoc)
            f_log.write('  Yoc: %s\n' % Yoc)
            if Yoc < Yr:
                Xs[-1] = Xoc
                Ys[-1] = Yoc
                continue
            else:
                pass #go Shrink
        #Inside Contraction
        elif Yr > Ys[-1]:
            Xic = Xc - params[2] * (Xr - Xc)
            Yic = func(Xic, *args)
            f_log.write('  Xic: %s\n' % Xic)
            f_log.write('  Yic: %s\n' % Yic)
            if Yic < Ys[-1]:
                Xs[-1] = Xic
                Ys[-1] = Yic
                continue
            else:
                pass #go Shrink

        #Shrink
        f_log.write('  Shrink\n')
        for i in range(1,n+1):
            Xs[i] = Xs[0] + params[3] * (Xs[i] - Xs[0])
        Ys = None #enforce to re-calculate

    #sorting
    sorted_idx = np.argsort(Ys)
    Xs = Xs[sorted_idx]
    Ys = Ys[sorted_idx]
    X_best = Xs[0]
    Y_best = Ys[0]

    f_log.write('Final result\n')
    f_log.write('  X_best: %s\n' % X_best)
    f_log.write('  Y_best: %s\n' % Y_best)
    return X_best, Y_best

def simplex_check_converge(Xs, Ys, x_tol=None, y_tol=None, y_abs=None):
    """For check simplex converge.

    :Xs: sorted x
    :Ys: sorted y
    :x_tol: converge if max x diff <= x_tol
    :y_tol: converge if max y diff <= y_tol
    :y_abs: converge if any y <= y_abs
    :returns: True or False

    """
    if y_abs:
        if Ys[0] <= y_abs:
            return True
    
    if y_tol:
        if Ys[-1] - Ys[0] <= y_tol:
            return True
    
    if x_tol:
        x_tol_conv = True
        x_min = np.amin(Xs, axis=0)
        x_max = np.amax(Xs, axis=0)
        for i in range(x_min.shape[0]):
            if x_max[i] - x_min[i] > x_tol[i]:
                x_tol_conv = False
        if x_tol_conv:
            return True

    return False

