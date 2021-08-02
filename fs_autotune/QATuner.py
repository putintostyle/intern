#!/usr/bin/env python3
import os
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


class QATunerBase:

    """QATuner base class. """

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
        raise NotImplementedError("QATunerBase: run function not implemented!")


class QASingleRunTuner(QATunerBase):

    """For run job 1 time"""

    def __init__(self, settings, runner):
        super().__init__(settings, runner)

    def run(self, arg1 = None):
        """main for run.

        :arg1: TODO
        :returns: TODO

        """
        print('Tuner start run')
        self.logger.info('Tuning... Start')
        if not os.path.isdir(self.result_folder):
            os.mkdir(self.result_folder)

        self.job_runner.set_cases(self.cases)
        self.job_runner.set_result_db(self.qa_db)
        self.job_runner.run()

        #show all layer statistics
        self.job_runner.print_statistics() #print on screen

        #write results to file
        self.job_runner.print_results("%s/QA_result_single_run.txt" % (self.result_folder))
        self.job_runner.print_statistics("%s/QA_result_single_run.statistics.txt" % (self.result_folder))
        
        #plot result
        if not self.settings.no_plot:
            plot_path = "%s/QA_result_single_run.png" % (self.result_folder)
            self.job_runner.plot_records(not self.settings.no_plot_show, plot_path)
        self.logger.info('Tuning... End')


class QAWidthTuner(QATunerBase):

    """QAWidthTuner for tuning line width. """

    def __init__(self, settings, runner):
        super().__init__(settings, runner)
        self.wext_fitter = None
        self._previous_cost = None
        self.itr_data_keys = [] #for accessing first run data
        self.resultsList = [] #only for output

    def run(self, arg1 = None):
        """main for run.

        :arg1: TODO
        :returns: TODO

        """
        print('Tuner start run')
        self.logger.info('Tuning... Start')
        if not os.path.isdir(self.result_folder):
            os.mkdir(self.result_folder)
        self.build_regression_data(None)
        self.calculate_expected_wext(None)
        self.output_wext_result()
        self.fit_wext_distribution(None)
        plot_result = not self.settings.no_plot
        show_plot_result = not self.settings.no_plot_show
        self.output_wext_fit_result(plot_result, show_plot_result)
        self.logger.info('Tuning... End')

    def cost(self, result):
        """cost function.

        :result: TODO
        :returns: TODO

        """
        #(c_all, m_all, val_all, cc_all, cm_all, cval_all)
        spec = 2.0*self.settings.cost_adjust
        return (result[2]/spec)**2 # normalize spec to 1


    def build_regression_data(self, arg1):
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

        f_log = open("%s/QA_wext_build_regression_data.log" % (self.result_folder), 'w')

        self.job_runner.set_cases(self.cases)
        self.job_runner.set_result_db(self.qa_db)
        # for itr
        for itr in range(2): #XXX
            if converge:
                break

            # run
            #self.job_runner.set_cases(self.cases)
            self.job_runner.settings.manual_tuning_param['wext'] = param
            params = self.job_runner.tuningParam2tup()
            keys = list(zip(self.cases, [params]*len(self.cases)))
            self.itr_data_keys.append(keys)
            #print(keys)
           
            print('-'*80)
            print("itr %s, wext = %s" % (itr, self.job_runner.settings.manual_tuning_param['wext']))
            f_log.write("itr %s, wext = %s\n" % (itr, self.job_runner.settings.manual_tuning_param['wext']))
            self.job_runner.run()
            self.job_runner.print_results("%s/QA_result_wext_itr_%s.txt" % (self.result_folder, itr))
            #show all layer statistics
            self.job_runner.print_statistics("%s/QA_result_wext_itr_%s.statistics.txt" % (self.result_folder, itr))
            #self.job_runner.plot_records()

            # get result
            # check converge

            cost = self.cost(self.job_runner.get_statistics())
            print("    normalized cost: %s" % (cost))
            print("    (count, mean, 2sigma, cccount, ccmean, cc2sigma) = %s" % str(self.job_runner.get_statistics()))
            print()
            f_log.write("normalized cost: %s\n" % (cost))
            f_log.write("(count, mean, 2sigma, cccount, ccmean, cc2sigma) = %s" % str(self.job_runner.get_statistics()))
            f_log.write('\n')

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

        f_log.close()

    def calculate_expected_wext(self, arg1):
        """calculate expected wext.

        :arg1: TODO
        :returns: TODO

        """
        self.logger.info('calculate_expected_wext')

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


class QAZTunerBase(QATunerBase):

    """QAZTuner base class for minimizing Ct or Cc error difference by adjust z. """

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
        print('Tuner start run')
        self.logger.info('Tuning... Start')
        if not os.path.isdir(self.result_folder):
            os.mkdir(self.result_folder)

        #self.example_method_tune_z(None)
        self.simplex_method_tune_z()
        
        self.logger.info('Tuning... End')

    def cost_function(self, result):
        """cost function
        """
        raise NotImplementedError("QAZTunerBase: cost function not implemented!")
            
            
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




class QACcCtDiffTuner(QAZTunerBase):

    """QACcCtDiffTuner for minimizing Ct Cc difference by adjust z. """

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


class QACtZTuner(QAZTunerBase):

    """QACtZTuner for minimizing Ct error by adjust z. """

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

