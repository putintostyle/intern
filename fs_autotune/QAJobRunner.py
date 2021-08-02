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

import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pandas as pd


from QAUtil import *
from QAResult import *


try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def progressbar(it, total=None, size=60, prefix="", file=sys.stdout):
    if total:
        count = total
    else:
        count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

try:
    import tqdm
except ImportError:
    pbar = progressbar
else:
    pbar = tqdm.tqdm

class QAJobReasultReader:
    def __init__(self, settings):
        self.settings = settings
        self.cases = []
        self.logger = logging.getLogger(type(self).__name__)
        self.qa_db = None
    
    def set_cases(self, cases):
        self.cases = cases

    def set_result_db(self, qa_db):
        self.qa_db = qa_db

    def tuningParam2tup(self, param = None):
        if param is None:
            param = self.settings.manual_tuning_param
        if 'post_tuning' in param:
            post_tune = True
        else:
            post_tune = False
        if param:
            tuning_params = (param['wext'], \
                             param['z0ext'], \
                             param['z1ext'], \
                             param['zloext'], \
                             param['zhiext'], \
                             post_tune \
                             )
        else:
            tuning_params = None
        return tuning_params
    
    def get_result(self, case_name, params):
        result = self.qa_db.get_result(case_name, params)
        return result
    
    #def get_results(self, cases): #XXX
    #    pendingTasks = cases[0]
    #    param = cases[1]
    #    while idx < len(pendingTasks):
    #        if(idx%10 == 0):
    #            print('%.1f%%' % (float(idx) / len(pendingTasks) *100.0))
    #        case_name = pendingTasks[idx]
    #        tuning_params = self.tuningParam2tup(param)
    #        result_list = self.get_result(case_name, tuning_params)
    #    return result_list
    
    def get_results(self):
        tuning_params = self.tuningParam2tup()
        result_list = self.qa_db.get_results(self.cases, tuning_params)
        return result_list

    def get_results_by_CDSP(self, CD1, CD2, SP1, SP2):
        tuning_params = self.tuningParam2tup()
        names, result_list = self.qa_db.get_results_by_CDSP_range(self.cases, tuning_params, CD1, CD2, SP1, SP2)
        return names, result_list



class QAJobRuner:
    def __init__(self, settings):
        self.settings = settings
        self.cases = []
        self.resultsList = []
        self.logger = logging.getLogger(type(self).__name__)
        self.qa_db = None 
    
    def set_cases(self, cases):
        self.cases = cases

    def set_result_db(self, qa_db):
        self.qa_db = qa_db

    def get_wxf(self, case_name):
        #cmd = self.qa_db.get_cmd(case_name)
        wxf = self.qa_db.get_CDSP(case_name)
        if not wxf:
            _, wxf = self.read_fslog()
        #self.qa_db.add_qa_data(case_name, Cmd=cmd, CD=wxf[0], SP=wxf[1]) # store to QA data file
        return wxf

    def read_fslog (self, fname='fs.log'):
        lines = []
        cmdOptions = None
        wxf = None

        with open(fname) as fp:
            lines = fp.readlines()
        for line in lines:
            ts = line.split()
            if len(ts) == 0:
                continue
            if ts[0] == 'Command':
                cmdOptions = ts[3:]
            elif ts[0] == 'wxf':
                #-- find width and spacing
                widx = 0
                sidx = 0
                for i in range(len(ts)):
                    if ts[i] == 'width':
                        widx = i+1
                    elif ts[i] == 'space':
                        sidx = i+1
                wxf = (int(ts[widx]), int(ts[sidx]))
        return (cmdOptions, wxf)


    def read_fsnets (self, fname='fs.nets'):
        nets = []
        with open(fname) as fp:
            lines = fp.readlines()
        for line in lines:
            ts = line.split()
            if ts and ts[0] == 'extract':
                nets.append(ts[1])
        return nets


    def read_fsrpt(self, fname='fs.report'):
        lines = []
        bstart = False

        with open(fname) as fp:
            lines = fp.readlines()

        ccs = {}
        tcs = {}
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            ts = line.split()
            if ts and line.startswith('Capacitance Report:'):
                idx += 4
                while True:
                    line = lines[idx]
                    ts = line.split()
                    if not ts:
                        break;
                    net1 = ts[0].split('$')[-1]
                    net2 = ts[1].split('$')[-1]
                    cc = float(ts[2])
                    if net1[0] != 'M':
                        net1, net2 = net2, net1
                    ccs[(net1, net2)] = cc
                    idx += 1
            elif ts and line.startswith('Report of Net-group'):
                idx += 4
                while True:
                    line = lines[idx]
                    ts = line.split()
                    if not ts:
                        break;
                    net1 = ts[0]
                    tc = float(ts[1])
                    tcs[net1] = tc
                    idx += 1
            else:
                idx += 1

        return (tcs, ccs)


    def read_golden(self, fname='golden'):
        lines = []
        with open(fname) as fp:
            lines = fp.readlines()

        tc = float(lines[0])
        cc = float(lines[1])
        return (tc, cc)
    
    def tuningParam2tup(self, param = None):
        if param is None:
            param = self.settings.manual_tuning_param
        if 'post_tuning' in param:
            post_tune = True
        else:
            post_tune = False
        if param:
            tuning_params = (param['wext'], \
                             param['z0ext'], \
                             param['z1ext'], \
                             param['zloext'], \
                             param['zhiext'], \
                             post_tune \
                             )
        else:
            tuning_params = None
        return tuning_params

    def make_exp_name(self, caseName, **kwargs):
        tuningStr = '#'.join( ['{}_{}'.format(k,v) for k, v in kwargs.items() ] )
        if tuningStr:
            expName = caseName + '#' + tuningStr
        else:
            expName = caseName
        return expName

    def get_result(self, case_name, params, wxf, net_pattern = None, skip_check_wxf = False):
        if net_pattern:
            result = self.get_result_from_file(wxf, net_pattern, skip_check_wxf)
            return result

        result = self.qa_db.get_result(case_name, params)
        #print(result)
        if not result:
            result = self.get_result_from_file(wxf, net_pattern, skip_check_wxf)
        #self.qa_db.add_qa_data(case_name, Cmd=cmd, CD=wxf[0], SP=wxf[1]) # store to QA data file
        return result

    def get_result_from_file(self, wxf, net_pattern = None, skip_check_wxf = False):
        if wxf == None:
            self.logger.info("wxf not found in fs.log, read from fs.log2.")
            _, wxf = self.read_fslog('fs.log2') #read from new run log
        if wxf == None and skip_check_wxf:
            self.logger.warning("wxf not found in fs.log/fs.log2, write dummy values.")
            wxf = (-100, -100) #dummy values

        rtup = None
        if net_pattern:
            rtup = self.get_err_by_pattern(net_pattern) + wxf
        else:
            rtup = self.get_err() + wxf
        return rtup



    def get_err_by_pattern(self, net_pattern):
        tcs, ccs = self.read_fsrpt()
        gtc, gcc = self.read_golden()
        nets = self.read_fsnets()
        if net_pattern != None:
            net1 = None
            cc = 0
            self.logger.warning("net_pattern specified. Direct detect net name from result.")
            nets_tmp = tcs.keys()
            net_n = 0
            tcerr = 0
            for net in nets_tmp:
                self.logger.debug('check net: %s' % net)
                if re.search( net_pattern, net):
                    net1 = net
                    tc = tcs[net1] * 1e15
                    tcerr += 100 * (tc-gtc)/gtc
                    net_n += 1
            if net_n > 0:
                tcerr = tcerr/net_n
            ccerr = 0
            return (tcerr, ccerr)
        else:
            return (-100, -100)

    def get_err(self):
        tcs, ccs = self.read_fsrpt()
        gtc, gcc = self.read_golden()
        nets = self.read_fsnets()

        if len(nets) == 1:
            net1 = nets[0]
            cc = 0
        elif len(nets) == 2: 
            net1, net2 = nets
            if net1[0] != 'M':
                net1, net2 = net2, net1
            if (net1,net2) in ccs:
                cc = ccs[net1, net2] * 1e15
            else:
                cc = 0
        elif len(nets) == 0:
            net1 = None
            cc = 0
            self.logger.info("empty fs.nets flow. Direct detect net name from result.")
            nets_tmp = tcs.keys()
            for net in nets_tmp:
                self.logger.debug('check net: %s' % net)
                if net[0] == 'M':
                    net1 = net
        else:
            net1 = None
            cc = 0
            for net in nets:
                self.logger.debug('check net: %s' % net)
                if net[0] == 'M':
                    net1 = net
        if net1 == None:
            tc = 0;
            self.logger.warning("net not found")
        else:
            tc = tcs[net1] * 1e15
        tcerr = 100 * (tc-gtc)/gtc
        if cc == 0 or gcc == 0:
            ccerr = -100
        else:
            ccerr = 100 * (cc-gcc)/gtc
        return (tcerr, ccerr)


    def clear_folder(self, path):
        #-- this seems unstable
        # shutil.rmtree(path)
        sp.run('rm -rf {}'.format( path ), shell=True)

    def cd_work_dir(self, path=None):
        cd_path = "work"
        if self.settings.manual_tune:
            cd_path = "work_tune"
        elif path:
            cd_path = path

        if not os.path.isdir( cd_path ):
            self.logger.error("Cannot cd to the folder %s!!" % (cd_path))
        else:
            os.chdir(cd_path)
        return cd_path

    # only for debug
    def one_dummy(self, name, local, force, skip_check_wxf, tunings, sleep_time = 0.1):
        mycwd = os.getcwd()
        expName = self.make_exp_name(name, **tunings)
        exeName = 'rapid3d' if local==True else 'qsub.py -m 20 --quiet --os7 -- rapid3d'
        self.logger.info('Running... {} local={} force={}'.format(name, local, force))
        rtup = None
        rtup = (-100.0, -100.0) +  (-100, -100) # dummy values
        time.sleep(sleep_time)
        return rtup

    def run(self):
        pendingTasks = list( zip( self.cases, [self.settings.force] * len(self.cases) ) )
        idx = 0
        skip_wxf = self.settings.skip_check_wxf
        net_pattern = self.settings.net_pattern
        self.resultsList = [] #clean result

        if self.settings.nopool:
            while idx < len(pendingTasks):
                if(idx%10 == 0):
                    print('%.1f%%' % (float(idx) / len(pendingTasks) *100.0))
                #--  run a job or a set of tuning jobs
                d, bForce = pendingTasks[idx]
                task = (d, self.settings.local, bForce, skip_wxf, net_pattern, self.settings.manual_tuning_param)
                result = self.one_experiment(task)
                name = result[0]
                tcerr, ccerr, aw, asp = result[1]
                self.resultsList.append( (name, tcerr, ccerr, aw, asp) )
                self.qa_db.add_qa_data(name, CD=aw, SP=asp)
                tuning_params = self.tuningParam2tup(self.settings.manual_tuning_param)
                self.qa_db.add_result(name, tuning_params, Ct_err = tcerr, Cc_err = ccerr, CD = aw, SP = asp)
                self.resultsList.sort()
                idx+=1
            self.qa_db.save()
        else:
            #-- run!
            expsMap = {}
            with mp.Pool(processes=self.settings.cpu_num) as pool:
                totalTasks = len(pendingTasks)
                
                tasks = []
                while idx < totalTasks:
                    #--  run a job or a set of tuning jobs
                    d, bForce = pendingTasks[idx]
                    tasks.append((d, self.settings.local, bForce, skip_wxf, net_pattern, self.settings.manual_tuning_param))
                    idx+=1
                
                result_list = []
                for R in pbar(pool.imap_unordered(self.one_experiment, tasks), total=len(tasks)):
                    result_list.append(R)

                for result in result_list:
                    name = result[0]
                    tcerr, ccerr, aw, asp = result[1]
                    self.resultsList.append( (name, tcerr, ccerr, aw, asp) )
                    self.qa_db.add_qa_data(name, CD=aw, SP=asp)
                    tuning_params = self.tuningParam2tup(self.settings.manual_tuning_param)
                    self.qa_db.add_result(name, tuning_params, Ct_err = tcerr, Cc_err = ccerr, CD = aw, SP = asp)
            self.resultsList.sort()
        self.qa_db.save()

    #def one_experiment(self, name, local, force, skip_check_wxf, net_pattern, tunings):
    def one_experiment(self, args):
        name, local, force, skip_check_wxf, net_pattern, tunings = args
        mycwd = os.getcwd()
        expName = self.make_exp_name(name, **tunings)
        exeName = 'rapid3d' if local==True else 'qsub.py -m 30 --quiet --os7 -- rapid3d'
        self.logger.info('Running... {} local={} force={}'.format(name, local, force))
        rtup = None
        current_dir = self.cd_work_dir()
        try:
            if os.path.isdir( expName ) and force is False:
                os.chdir(expName)
                wxf = self.get_wxf(name)
                
                # get result
                tuning_params = self.tuningParam2tup(self.settings.manual_tuning_param)
                rtup = self.get_result(name, tuning_params, wxf, net_pattern, skip_check_wxf)

            else:
                if os.path.isdir( expName ):
                    self.clear_folder(expName)
                os.mkdir( expName )
                os.chdir( expName )
                sp.run('cp ../../data/{}/* .'.format(name), shell=True)
                cmdOptions, wxf = self.read_fslog()

                #-- turn the tunings to a flattened list
                tuningSettings = list(itertools.chain.from_iterable( [ ('-' + k, str(v)) for k,v in tunings.items() ] ))
                opts = cmdOptions + tuningSettings
                s1 = sp.run( ' '.join([exeName] + opts), shell=True, stdout=sp.PIPE, stderr=sp.STDOUT, universal_newlines=True)
                self.logger.debug(' '.join([exeName] + opts))
                s2 = sp.run('rapid3d -bfs2ascii fs.report fs.report', shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)

                with open('fs.log2', 'w') as logfp:
                    logfp.write('-'*100 + '\n')
                    logfp.write(s1.stdout)
                    logfp.write('-'*100 + '\n')

                # get result
                rtup = self.get_result_from_file(wxf, net_pattern, skip_check_wxf)
                
        except Exception as e:
            self.logger.error('Exception raised in folder %s' % os.path.join(current_dir, expName))
            self.logger.error('Write error message to file %s' % (os.path.join(current_dir, expName, 'err.log')))
            with open('err.log', 'w') as fp:
                fp.write(str(e) + '\n')
                tb = traceback.format_exc()
                fp.write(str(tb) + '\n')
            pass

        os.chdir(mycwd)
        return (name, rtup)


    def post_process(self):
        resultsList = self.resultsList
        pass


    def print_results(self, filename = None):
        self.logger.info('Print results') 
        if filename:
            print('\n    Write results to %s' % filename)
            self.logger.info('Write results to %s' % filename)

        resultsList = self.resultsList

        if filename and filename != "":
            fh = open(filename, 'w')
        else:
            fh = sys.stdout

        #--- find mean
        N = len(resultsList)
        tcErrTotal =  sum( t[1] for t in resultsList )
        tcmean = tcErrTotal / N

        ccErrTotal =  sum( t[2] for t in resultsList )
        ccmean = ccErrTotal / N

        #--  find stderr
        tcerrsq = 0
        ccerrsq = 0

        print('-'*100, file = fh)
        print("   Name,    vs,    Ct Error,    Cc Error,     CD,     SP", file = fh)
        print('-'*100, file = fh)
        prevName = ''
        for name0, tcerr, ccerr, aw, asp in resultsList:
            tcerrsq += tcerr ** 2
            ccerrsq += ccerr ** 2

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
            fmts.extend([ '{:7.2f}', '{:7.2f}', '{:7.1f}', '{:7.1f}'])
            fmtStr = ' '.join(fmts)
            print(fmtStr.format(name, *vs, tcerr, ccerr, aw, asp), file = fh)

        #-- print final result
        print('-'*100, file = fh)
        print('Tc err mean/2sigma: {:6.2f} {:6.2f}'.format( tcmean, 2 * math.sqrt( tcerrsq/N ) ), file = fh)
        print('Cc err mean/2sigma: {:6.2f} {:6.2f}'.format( ccmean, 2 * math.sqrt( ccerrsq/N ) ), file = fh)
        print('-'*100, file = fh)

    def print_statistics(self, filename = None):
        self.logger.info('Print layer statistics results')
        if self.settings.skip_check_name:
            self.logger.info('Skip print due to --skip_check_name')
            return None

        if filename:
            print('    Write layer statistics results to %s' % filename)
            self.logger.info('Write layer statistics results to %s' % filename)
        else:
            print('\nPrint layer statistics results')

        resultsList = self.resultsList
        if filename and filename != "":
            fh = open(filename, 'w')
        else:
            fh = sys.stdout

        statics = {}
        prevName = ''
        for name0, tcerr, ccerr, aw, asp in resultsList:
            ts = name0.split('#')
            name = ts[0] 
            if name != prevName:
                # print('-'*40)
                prevName = name
            header = name.replace('M',',M')[1:].split(',')
            layer_id = int(header[0].replace('M',''))

        #for k, r in self.raw_result.items():
            if layer_id in statics:
                statics[layer_id][0] += 1
                statics[layer_id][1] += tcerr
                statics[layer_id][2] += tcerr**2
                if ccerr > -100:
                    statics[layer_id][3] += 1
                    statics[layer_id][4] += ccerr
                    statics[layer_id][5] += ccerr**2
            else:
                if ccerr > -100:
                    statics[layer_id] = [1, tcerr, tcerr**2, 1, ccerr, ccerr**2]
                else:
                    statics[layer_id] = [1, tcerr, tcerr**2, 0, 0, 0]

        print('-'*80, file = fh)
        print('Layer : count  mean  2-sigma c_count  c_mean  c_2-sigma', file = fh)
        for id in sorted( statics.keys() ):
            c, m, val, cc, cm, cval, = statics[id]
            if c == 0: # prevent devide zero
                c_ = 1
            else:
                c_ = c
            if cc == 0:
                cc_ = 1
            else:
                cc_ = cc
            print( 'M{} : {}  {:+.2f}  {:.2f}  {}  {:+.2f}  {:.2f}'.format(id, c, m/c_, 2.0*math.sqrt(val/c_), cc, cm/cc_, 2.0*math.sqrt(cval/cc_)), file = fh) #layer, count, mean diff, 2-sigma
        print('-'*80, file = fh)

    def get_statistics(self):
        self.logger.info('get statistics results')
        if self.settings.skip_check_name:
            self.logger.info('Skipped due to --skip_check_name')
            return None

        resultsList = self.resultsList
        #if filename and filename != "":
        #    fh = open(filename, 'w')
        #else:
        #    fh = sys.stdout

        statics = {}
        prevName = ''
        for name0, tcerr, ccerr, aw, asp in resultsList:
            ts = name0.split('#')
            name = ts[0] 
            if name != prevName:
                # print('-'*40)
                prevName = name
            header = name.replace('M',',M')[1:].split(',')
            layer_id = int(header[0].replace('M',''))

        #for k, r in self.raw_result.items():
            if layer_id in statics:
                statics[layer_id][0] += 1
                statics[layer_id][1] += tcerr
                statics[layer_id][2] += tcerr**2
                if ccerr > -100:
                    statics[layer_id][3] += 1
                    statics[layer_id][4] += ccerr
                    statics[layer_id][5] += ccerr**2
            else:
                if ccerr > -100:
                    statics[layer_id] = [1, tcerr, tcerr**2, 1, ccerr, ccerr**2]
                else:
                    statics[layer_id] = [1, tcerr, tcerr**2, 0, 0, 0]

        #print('-'*80, file = fh)
        #print('Layer : count  mean  2-sigma c_count  c_mean  c_2-sigma', file = fh)
        #for id in sorted( statics.keys() ):
        #    c, m, val, cc, cm, cval, = statics[id]
        #    if c == 0:
        #        c = 1
        #    if cc == 0:
        #        cc = 1
        #    print( 'M{} : {}  {:+.2f}  {:.2f}  {}  {:+.2f}  {:.2f}'.format(id, c, m/c, 2.0*math.sqrt(val/c), cc, cm/cc, 2.0*math.sqrt(cval/cc)), file = fh) #layer, count, mean diff, 2-sigma
        #print('-'*80, file = fh)
        
        c_all = 0
        m_all = 0.0
        val_all = 0.0
        cc_all = 0
        cm_all = 0.0
        cval_all = 0.0
        for id in sorted( statics.keys() ):
            c, m, val, cc, cm, cval, = statics[id]
            
            c_all += c
            m_all += m
            val_all += val
            cc_all += cc
            cm_all += cm
            cval_all = cval
        if c_all == 0:
            c_all_ = 1 #prevent devide zero
        else:
            c_all_ = c_all
        if cc_all == 0:
            cc_all_ = 1
        else:
            cc_all_ = cc_all
        m_all = m_all/c_all_
        val_all = 2.0*math.sqrt(val_all/c_all_)
        cm_all = cm_all/cc_all_
        cval_all = 2.0*math.sqrt(cval_all/cc_all_)
        return (c_all, m_all, val_all, cc_all, cm_all, cval_all)


    def plot_records(self, show_plot=True, save_path="QA_result.png"):
        self.logger.info('Plot results') 
        print()
        print('Plot results') 
        print('-'*80)

        result = self.resultsList

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

        for name, tcerr, ccerr, aw, asp in result:
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



