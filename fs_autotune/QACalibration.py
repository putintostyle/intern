#!/usr/bin/env python3


# from typing import runtime_checkable
class Calibrator:

    """Docstring for Calibrator. """

    def __init__(self):
        """TODO: to be defined. """
        self.rule_set_list = [] #XXX
        self.segment_rule = []
    def apply(self, arg1):
        """TODO: Docstring for apply.

        :arg1: TODO
        :returns: TODO

        """
        pass

    def check_rule(self, cd, sp): #XXX: need to update input to an object
        """TODO: Docstring for check_rule.

        :arg1: TODO
        :returns: TODO

        """
        results = []
        for s in self.rule_set_list:
            r = s.check_rule(cd, sp)
            results.append(r)
            #print(r)
        
        final_dw = 0
        for r in results:
            final_dw += r #XXX
        return final_dw

    def add_rule_from_str(self, data_str):
        """TODO: Docstring for add_rule_from_str.

        :data_str: TODO
        :returns: TODO

        """
        if self.rule_set_list == []:
            cal_creator = CalRuleSetBuilder()
            cal_set = cal_creator.build(None)
            self.rule_set_list.append(cal_set)
        else:
            cal_set = self.rule_set_list[0] #XXX
        cal_set.add_rule_from_str(data_str)
        #print(len(self.rule_set_list))

    def add_rule_set(self, cal_rule_set):
        """TODO: Docstring for add_rule_set.

        :arg1: TODO
        :returns: TODO

        """
        self.rule_set_list.append(cal_rule_set)

    def get_rule_set(self, arg1):
        """TODO: Docstring for get_rule_set.

        :arg1: TODO
        :returns: TODO

        """
        pass

    def clean(self):
        """TODO: Docstring for clean.
        :returns: TODO

        """
        self.rule_set_list = [] #XXX

    def to_file(self, arg1):
        """TODO: Docstring for to_file.

        :arg1: TODO
        :returns: TODO

        """
        pass

    def print_all(self, arg1 = None):
        """TODO: Docstring for print_all.

        :arg1: TODO
        :returns: TODO

        """
        for rs in self.rule_set_list:
            print("rule set:")
            rs.print_all()
    # rule_set_list contains only constructor

    def print_region_cond(self, region): # region = [CD1, CD2, SP1, SP2]
        self.segment_rule = []
        for rs in self.rule_set_list:
            ## check is overlapped or not
            for rule in rs.rules:
                ## rule = {'CD1': 'CD2': 'SP1': 'SP2': 'dW':}
                if (rule.data['CD1']<= region[1]) & (rule.data['CD2']>= region[0]) & (rule.data['SP1']<= region[3]) & (rule.data['SP2']>= region[2]):
                    CD1, CD2 = (max(int(rule.data['CD1']), int(region[0])), min(int(rule.data['CD2']), int(region[1])))
                    SP1, SP2 = (max(int(rule.data['SP1']), int(region[2])), min(int(rule.data['SP2']), int(region[3])))
                    self.segment_rule.append([CD1, CD2, SP1, SP2, rule['dW']])
                    print('{} <= CD <= {}, {} <= SP <= {}, wext = {}'.format(CD1, CD2, SP1, SP2, rule['dW']))
                    ## need to do：print rules
                    ## need to do：show region cal result
                    ## need to do：plot region 

class CalRuleSetBuilder:

    """Docstring for CalRuleSetBuilder. """

    def __init__(self):
        """TODO: to be defined. """
        #self.cal_rule_sets = []

    def build(self, arg1):
        """TODO: Docstring for build.

        :arg1: TODO
        :returns: TODO

        """
        rule_set = CalibrationRuleSet()
        return rule_set

class CalibrationRuleSet:

    """Docstring for CalibrationRuleSet. """

    def __init__(self):
        """TODO: to be defined. """
        self.filter = None
        self.rules = []
        self.rule_type = "Undefined" #XXX:change to enum

    def add_rule(self, arg1):
        """TODO: Docstring for add_rule.

        :arg1: TODO
        :returns: TODO

        """
        pass

    def add_rule_from_str(self, data_str):
        """TODO: Docstring for add_rule_from_str.

        :data_str: TODO
        :returns: TODO

        """
        cal_creator = CalRuleFactory()
        rule = cal_creator.create(data_str)
        self.rules.append(rule)

    def check_rule(self, cd, sp): #need to update
        """TODO: Docstring for check_rule.

        :arg1: TODO
        :returns: TODO

        """
        #XXX: need to update
        results = []
        for r in self.rules:
            result = r.check_rule(cd, sp)
            results.append(result)

        final_dw = 0
        for r in results:
            if r:
                final_dw += r #XXX
        return final_dw


    def apply(self, arg1):
        """TODO: Docstring for apply.

        :arg1: TODO
        :returns: TODO

        """
        pass

    def print_all(self, arg1 = None):
        """TODO: Docstring for print_all.

        :arg1: TODO
        :returns: TODO

        """
        for r in self.rules:
            print("  %s" % r)
        
    def to_file(self, arg1):
        """TODO: Docstring for to_file.

        :arg1: TODO
        :returns: TODO

        """
        pass

    def from_file(self, arg1):
        """TODO: Docstring for from_file.

        :arg1: TODO
        :returns: TODO

        """
        pass

    def _parsing_header(self, arg1):
        """TODO: Docstring for _parsing_head.

        :arg1: TODO
        :returns: TODO

        """
        pass

    def _output_header(self, arg1):
        """TODO: Docstring for _output_head.

        :arg1: TODO
        :returns: TODO

        """
        pass


class CalRuleFactory:

    """Docstring for CalRuleSetBuilder. """

    def __init__(self):
        """TODO: to be defined. """
        pass

    def create(self, data_str):
        """TODO: Docstring for build.

        :arg1: TODO
        :returns: TODO

        """
        rule_type = data_str.strip().split()[0]
        if rule_type == "CDSPW":
            rule = CalWRule()
            rule.from_str(data_str)
        else:
            rule = None
        return rule


class CalibrationRuleBase:

    """Docstring for CalibrationRuleBase. """

    def __init__(self):
        """TODO: to be defined. """
        self.rule_type = "Undefined" #XXX:change to enum

    def to_str(self):
        pass

    def from_str(self, data_str):
        pass

    def apply(self, arg1):
        pass
        #XXX: ???
        

class CalWRule(CalibrationRuleBase):

    """Docstring for CalWRule. """

    def __init__(self):
        """TODO: to be defined. """
        CalibrationRuleBase.__init__(self)
        self.rule_type = "CDSPW"
        self.data = {}

    def __str__(self):
        """TODO: Docstring for __str__.
        :returns: TODO

        """
        return self.to_str()

    def to_str(self):
        """TODO: Docstring for to_str.
        :returns: TODO

        """
        return "%s %s %s %s %s %s" % (self.rule_type, self.data['CD1'], self.data['CD2'], self.data['SP1'], self.data['SP2'], self.data['dW'])

    def from_str(self, data_str):
        """TODO: Docstring for from_str.

        :data_str: TODO
        :returns: TODO

        """
        data = data_str.strip().split()
        #XXX: check number
        self.data['CD1'] = int(data[1].strip())
        self.data['CD2'] = int(data[2].strip())
        self.data['SP1'] = int(data[3].strip())
        self.data['SP2'] = int(data[4].strip())
        self.data['dW']  = int(data[5].strip())

    def check_rule(self, cd, sp):
        """TODO: Docstring for check_rule.

        :: TODO
        :returns: TODO

        """
        #print("XXX: %s %s" % (cd, sp))
        #print("XXX data: %s %s %s %s %s" % (self.data['CD1'], self.data['CD2'], self.data['SP1'], self.data['SP2'], self.data['dW']))
        if self.data['CD1'] <= cd and cd < self.data['CD2'] and \
            self.data['SP1'] <= sp and sp < self.data['SP2']:
            return self.data['dW']
        else:
            return None

    #def get_rule_by_CDSP(self, cd1, cd2, sp1, sp2):
    #    """TODO: Docstring for get_rule_by_CDSP.

    #    :cd1: TODO
    #    :cd2: TODO
    #    :sp1: TODO
    #    :sp2: TODO
    #    :returns: TODO

    #    """
    #    pass
