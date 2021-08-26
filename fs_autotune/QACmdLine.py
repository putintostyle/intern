
import cmd
import string, sys
import csv

from numpy.lib.function_base import select
# from typing import runtime_checkable
from QAAnalyzer import *
from QACalibration import Calibrator
import os

class QAShellBase(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.prompt = "> "
        self.intro  = "input command:\n\te.g. q or r 100 200 100 200\n\te.g. ?[<cmd>]: help\n\te.g. !<cmd>: exec. shell command"  ## defaults to None

    ## Command definitions ##
    def do_hist(self, args):
        """Print a list of commands that have been entered"""
        print(self._hist)

    def do_exit(self, args):
        """Exits from the console"""
        return -1

    ## Command definitions to support Cmd object functionality ##
    def do_EOF(self, args):
        """Exit on system end of file character"""
        return self.do_exit(args)

    def do_shell(self, args):
        """Pass command to a system shell when line begins with '!'"""
        os.system(args)

    def do_help(self, args):
        """Get help on commands
           'help' or '?' with no arguments prints a list of commands for which help is available
           'help <command>' or '? <command>' gives help on <command>
        """
        ## The only reason to define this method is for the help text in the doc string
        cmd.Cmd.do_help(self, args)

    ## Override methods in Cmd object ##
    def preloop(self):
        """Initialization before prompting user for commands.
           Despite the claims in the Cmd documentaion, Cmd.preloop() is not a stub.
        """
        cmd.Cmd.preloop(self)   ## sets up command completion
        self._hist    = []      ## No history yet
        self._locals  = {}      ## Initialize execution namespace for user
        self._globals = {}

    def postloop(self):
        """Take care of any unfinished business.
           Despite the claims in the Cmd documentaion, Cmd.postloop() is not a stub.
        """
        cmd.Cmd.postloop(self)   ## Clean up command completion
        print("Exiting...")

    def precmd(self, line):
        """ This method is called after the line has been input but before
            it has been interpreted. If you want to modifdy the input line
            before execution (for example, variable substitution) do it here.
        """
        self._hist += [ line.strip() ]
        return line

    def postcmd(self, stop, line):
        """If you want to stop the console, return something that evaluates to true.
           If you want to do some post command processing, do it here.
        """
        return stop

    def emptyline(self):    
        """Do nothing on empty input line"""
        pass

    def default(self, line):       
        """Called on an input line when the command prefix is not recognized.
           In that case we execute the line as Python code.
        """
        try:
            exec(line) in self._locals, self._globals
        except Exception as e:
            print(e.__class__, ":", e)

    # shortcuts
    do_quit = do_exit
    do_q = do_quit
    do_h = do_hist



class QAAnalyzerShell(QAShellBase):

    """Docstring for QAAnalyzerShell. """

    def __init__(self, qa_analyzer):
        """TODO: to be defined. """
        QAShellBase.__init__(self)
        self.qa_analyzer = qa_analyzer
        self.select = None

    def do_global(self, args, calibration=False):
        """Select all cases. e.g. g [wext]"""
        self.qa_analyzer.settings.CDSP_range = False
        
        cmds = args.split()
        if len(cmds) > 0:
            self.qa_analyzer.settings.CDSP_range_param['wext'] = int(cmds[0].strip())
        else:
            self.qa_analyzer.settings.CDSP_range_param['wext'] = None
        self.qa_analyzer.run(calibration)
    
    def do_gc(self, args):
        """Select all cases with applying calibration. e.g. gc"""
        if self.manulcal:
            print('the stat in selected area is\n')
            selectRegion = str(int(float(self.select[0])))+' '+str(int(float(self.select[1])))+' '+str(int(float(self.select[2])))+' '+str(int(float(self.select[3])))
            self.do_range(self.select, calibration=True)
        else:
            self.do_global(args, calibration=True)

    def do_range(self, args, calibration=False):
        """Select cases in width(CD)/space(SP) range. e.g. r <CD1> <CD2> <SP1> <SP2> [wext]"""
        cmds = args.split()
        if len(cmds) < 4:
            print("Error: not enough args!")
            return
        self.qa_analyzer.settings.CDSP_range = True
        self.qa_analyzer.settings.CDSP_range_param['CD1'] = int(cmds[0].strip())
        self.qa_analyzer.settings.CDSP_range_param['CD2'] = int(cmds[1].strip())
        self.qa_analyzer.settings.CDSP_range_param['SP1'] = int(cmds[2].strip())
        self.qa_analyzer.settings.CDSP_range_param['SP2'] = int(cmds[3].strip())
        if len(cmds) > 4:
            self.qa_analyzer.settings.CDSP_range_param['wext'] = int(cmds[4].strip())
        else:
            self.qa_analyzer.settings.CDSP_range_param['wext'] = None
        self.qa_analyzer.run(calibration)

    def do_rc(self, args):
        """Select cases in width(CD)/space(SP) range with applying calibration. e.g. rc <CD1> <CD2> <SP1> <SP2>"""
        self.do_range(args, calibration=True)

    def do_cal(self, args):
        """Specify/update calibration rule.\n\te.g. cal <CD1> <CD2> <SP1> <SP2> <wext>\n\te.g. cal clean"""
        cmds = args.split()
        
        if len(cmds) < 1:
            print("Error: not enough args!")
            return

        if not self.qa_analyzer.calibrator:#XXX
            self.qa_analyzer.calibrator = Calibrator()

        cmd_1 = cmds[0].strip()
        if cmd_1=="clean":
            self.qa_analyzer.calibrator.clean()
            self.qa_analyzer.calibrator = None #XXX
        elif cmd_1=="show":
            self.qa_analyzer.calibrator.print_all()
        elif cmd_1=="opt": #get from last opt
            if 'CD1' not in self.qa_analyzer.settings.CDSP_range_param or not self.qa_analyzer.opt_wext:
                print("Error: no optimized wext!")
                return
            cmds_tmp = []
            cmds_tmp.append(self.qa_analyzer.settings.CDSP_range_param['CD1'])
            cmds_tmp.append(self.qa_analyzer.settings.CDSP_range_param['CD2'])
            cmds_tmp.append(self.qa_analyzer.settings.CDSP_range_param['SP1'])
            cmds_tmp.append(self.qa_analyzer.settings.CDSP_range_param['SP2'])
            cmds_tmp.append(self.qa_analyzer.opt_wext)
            rule_data_str = "%s %s %s %s %s %s" % ("CDSPW", cmds_tmp[0], cmds_tmp[1], cmds_tmp[2], cmds_tmp[3], cmds_tmp[4])
            self.qa_analyzer.calibrator.add_rule_from_str(rule_data_str)
        elif cmd_1=="save":
            pass #save to file
        elif cmd_1=="load":
            pass #load from file
        else:
            if len(cmds) < 5:
                print("Error: not enough args!")
                return
            rule_data_str = "%s %s %s %s %s %s" % ("CDSPW", cmds[0], cmds[1], cmds[2], cmds[3], cmds[4])
            self.qa_analyzer.calibrator.add_rule_from_str(rule_data_str)

    def do_plot(self, args):
        """Plot last results"""
        self.qa_analyzer.plot_last_result()
    def do_calfile(self, args):
        # usage: calfile filename
        if not self.qa_analyzer.calibrator:
            self.qa_analyzer.calibrator = Calibrator()
        # print(self.qa_analyzer.settings.layer)
        current_path = os.path.join(os.path.abspath(os.getcwd()),'__tuning__')
        cmds = args.split()
        if (len(cmds) == 1)&('.csv' in cmds[0]):
            file_name = cmds[0]
            rules_from_file = []
            with open(os.path.join(current_path, file_name), 'r', newline='') as file:
                rows = csv.reader(file)
                for row in rows:
                   rules_from_file.append(row)     
            header = rules_from_file[0]
            if ['CD1', 'CD2', 'SP1', 'SP2', 'wext'] == header:
                rule_array = rules_from_file[1:]
                for rule in rule_array:
                    rule_data_str = "%s %s %s %s %s %s" % ("CDSPW", int(float(rule[0])), int(float(rule[1])), int(float(rule[2])), int(float(rule[3])), int(float(rule[4])))
                    self.qa_analyzer.calibrator.add_rule_from_str(rule_data_str)
            else:
                print('wrong header')
        else:
            if (len(cmds) == 0):
                print("require file name")
            elif ('.csv' not in cmds[0]):
                print("require csv file")
    def do_autotune(self, args):
        # usage: -f file_name(auto), -t tree_number -ms 
        cmds = args.split()
        layer = self.qa_analyzer.settings.layer[0]
        sram = self.qa_analyzer.settings.sram[0]
        current_path = os.path.join(os.path.abspath(os.getcwd()))
        if ('-ms' in cmds ) & (self.select == None):
            print('please select a region with using mancal')
        else:
            if '-f' in cmds:
                file_operation = cmds[cmds.index('-f')+1]
                if file_operation == 'auto':
                    
                    if '-t' in cmds:
                        tree_number = cmds[cmds.index('-t')+1]
                        cmdstring = 'region_extraction.py -l {} -r {} --case_file_name -tn {} -wd {}'
                        if '-ms' in cmds:
                            os.system( (cmdstring+' --manul -c {} {} {} {}').format(layer, sram, tree_number, os.path.abspath(os.getcwd()), self.select[0], self.select[1], self.select[2], self.select[3]))
                        else:
                            os.system( (cmdstring).format(layer, sram, tree_number, os.path.abspath(os.getcwd())))
                    else:
                        cmdstring = 'region_extraction.py -l {} -r {} --case_file_name -wd {}'
                        if '-ms' in cmds:
                            os.system( (cmdstring+' --manul -c {} {} {} {}').format(layer, sram, os.path.abspath(os.getcwd()), self.select[0], self.select[1], self.select[2], self.select[3]))
                        else:
                            os.system( (cmdstring).format(layer, sram, os.path.abspath(os.getcwd())))

                else:
                    
                    if '-t' in cmds:
                        tree_number = cmds[cmds.index('-t')+1]
                        cmdstring = 'region_extraction.py -l {} -r {} --manul_file_name {} -tn {} -wd {}'
                        if '-ms' in cmds:
                            os.system( (cmdstring+' --manul -c {} {} {} {}').format(layer, sram, file_operation, tree_number, os.path.abspath(os.getcwd()), self.select[0], self.select[1], self.select[2], self.select[3]))
                        else:
                            os.system( cmdstring.format(layer, sram, file_operation, tree_number, os.path.abspath(os.getcwd())))
                    else:
                        cmdstring = 'region_extraction.py -l {} -r {} --manul_file_name {} -wd {}'
                        if '-ms' in cmds:
                            os.system( (cmdstring+' --manul -c {} {} {} {}').format(layer, sram, file_operation, os.path.abspath(os.getcwd()), self.select[0], self.select[1], self.select[2], self.select[3]))
                        else:
                            os.system(cmdstring.format(layer, sram, file_operation, os.path.abspath(os.getcwd())))
            else:
                if '-t' in cmds:
                    tree_number = cmds[cmds.index('-t')+1]
                    cmdstring = 'region_extraction.py -l {} -r {} -tn {} -wd {}'
                    if '-ms' in cmds:
                        os.system( (cmdstring+' --manul -c {} {} {} {}').format(layer, sram, tree_number, os.path.abspath(os.getcwd()), self.select[0], self.select[1], self.select[2], self.select[3]))
                    else:
                        os.system( cmdstring.format(layer, sram, tree_number, os.path.abspath(os.getcwd())))
                else:
                    cmdstring = 'region_extraction.py -l {} -r {} -tn {} -wd {}'
                    if '-ms' in cmds:
                        os.system((cmdstring+' --manul -c {} {} {} {}').format(layer, sram, 20, os.path.abspath(os.getcwd()), self.select[0], self.select[1], self.select[2], self.select[3]))
                    else:
                        os.system(cmdstring.format(layer, sram, 20, os.path.abspath(os.getcwd())))

    def do_autocal(self,args):
        if not self.qa_analyzer.calibrator:
            self.qa_analyzer.calibrator = Calibrator()
        current_path = os.path.join(os.path.abspath(os.getcwd()), '__tuning__')
        rules_from_file = []
        with open(os.path.join(current_path, 'rules.csv'), 'r', newline='') as file:
            rows = csv.reader(file)
            for row in rows:
                rules_from_file.append(row)     
        header = rules_from_file[0]
        if ['CD1', 'CD2', 'SP1', 'SP2', 'wext'] == header:
            rule_array = rules_from_file[1:]
            for rule in rule_array:
                rule_data_str = "%s %s %s %s %s %s" % ("CDSPW", int(float(rule[0])), int(float(rule[1])), int(float(rule[2])), int(float(rule[3])), int(float(rule[4])))
                self.qa_analyzer.calibrator.add_rule_from_str(rule_data_str)
        else:
            print('wrong header')
    def do_readcal(self, args):
        # usage: readcal CD1 CD2 SP1 SP2
        cmds = [int(i) for i  in args.split()]
        self.qa_analyzer.calibrator.specify_region(cmds, isprint=True)
        self.qa_analyzer.calibrator.clean_region()
    # shortcuts

    def do_regioncal(self, args):
        #usage: regioncal [-print] region
        cmds = [int(i) for i  in args.split()]
        
        self.qa_analyzer.calibrator.specify_region(cmds[-4:], isprint=('-print' in cmds))
                
        for seg_region in self.qa_analyzer.calibrator.segment_rule:
            self.qa_analyzer.settings.CDSP_range = True
            self.qa_analyzer.settings.CDSP_range_param['CD1'] = int(seg_region[0])
            self.qa_analyzer.settings.CDSP_range_param['CD2'] = int(seg_region[1])
            self.qa_analyzer.settings.CDSP_range_param['SP1'] = int(seg_region[2])
            self.qa_analyzer.settings.CDSP_range_param['SP2'] = int(seg_region[3])
            self.qa_analyzer.settings.CDSP_range_param['wext'] = None
            print('='*80)
            print('region in {} <= CD <= {}, {} <= SP <= {}, wext = {}'.format(seg_region[0], seg_region[1], seg_region[2], seg_region[3], seg_region[4]))
            self.qa_analyzer.region_run()
        print('='*80)
        self.qa_analyzer.calibrator.clean_region()
    


    def do_regionsel(self, args):
        ## usage:regionsel -os [select from origin] -or [select read stat from origin ] -ls [select form last cal]-lr [select read stat from last cal read]  
        ## show origin
        # return region
        # show statistic
        cmds =  args.split()
        if len(cmds) > 1:
            print('too  many args')
        os_flag = '-os' in cmds
        or_flag = '-or' in cmds
        ls_flag = '-ls' in cmds
        lr_flag = '-lr' in cmds
        if os_flag | ls_flag:
            select_region = self.qa_analyzer.plot(init = (os_flag|or_flag) , select = True, read = False)
        else:
            self.qa_analyzer.plot(init = (os_flag|or_flag) , select = False, read = True)
        print(select_region)
        print('closed selector')
    def do_mancal(self, args):
        # usage: mancal 
        cmds = args.split()
        self.manulcal = True
        self.select = self.qa_analyzer.plot(init = True , select = True, read = False)
        


         
    do_g = do_global
    do_r = do_range
    do_p = do_plot
    do_f = do_calfile
