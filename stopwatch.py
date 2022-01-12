

import re
from time import process_time

class stopwatch(object):
    timestamps = []
    names      = {}
    iteration  = {}
    _timer     = process_time
    verbosity  = 0

    def __init__(self,timer=None):
        if timer is not None:
            self._timer = timer
        self.stamp("Instantiation")
        return

    def add_name(self,name):
        if name not in self.names.keys():
            self.iteration[name] = 0
            self.names[name] = len(self.timestamps)
        else:
            self.iteration[name] = self.iteration[name] + 1
            self.names["%s(%i)"%(name,self.iteration[name])] = len(self.timestamps)

    def stamp(self,name=None):
        if name is None:
            name = "stamp-%s"%len(self.names.keys())
        self.add_name(name)
        self.timestamps.append(self._timer())
        if self.verbosity > 0:
            print(self.report())
        return self.timestamps[-1]

    def delta(self,name1=None,name2=None):
        if name1 is None and name2 is None:
            return self.timestamps[-1] - self.timestamps[0]
        if name2 is None:
            return self.timestamps[-1] -  self.timestamps[self.names[name1]]
        if name1 is None:
            return self.timestamps[name2] -  self.timestamps[0]
        return self.timestamps[self.names[name2]] - self.timestamps[self.names[name1]]

    def current(self):
        return self._timer()

    def delta_since_start(self):
        return self.current() - self.timestamps[0]

    def report(self):
        keys = list(self.names.keys())
        return self.report_string("Instantiation",keys[-1])

    def report_string(self,name1,name2,message=None):
        if message is None:
            message = "'%s','%s',"%(name1,name2)
        return message+"%06e"%self.delta(name1,name2)

    def report_all(self,prefix=""):
        ret_str = ""
        keys = list(self.names.keys())
        for i in range(len(self.timestamps)):
            ret_str=ret_str+prefix+self.report_string("Instantiation",keys[i])+"\n"
        return ret_str

    def summary(self,prefix=''):
        keys = list(self.names.keys())
        deltas = []
        deltas_gather = {}
        for i in range(1,len(self.timestamps)):
            deltas.append(self.timestamps[i]-self.timestamps[i-1])
            m = re.search('^(.*)\((.*)\)$',keys[i])
            if m is None:
                deltas_gather[keys[i]] = [deltas[-1]]
            else:
                deltas_gather[m.group(1)].append(deltas[-1])
        keys_gather = list(deltas_gather.keys())
        totals = {}
        iters  = {}
        ret_str = prefix + '<stopwatch-summary>\n'
        ret_str = ret_str + prefix + "'key','sum (s)','min','max','iterations'\n"
        for i in range(0,len(keys_gather)):
            totals[keys_gather[i]] = sum(deltas_gather[keys_gather[i]])
            iters[keys_gather[i]]  = len(deltas_gather[keys_gather[i]])
            ret_str = ret_str + prefix + "'%s',%f,%f,%f,%i\n"\
                      %(keys_gather[i]\
                        ,totals[keys_gather[i]]\
                        ,min(deltas_gather[keys_gather[i]])\
                        ,max(deltas_gather[keys_gather[i]])\
                        ,iters[keys_gather[i]])
        ret_str = ret_str + prefix + '</stopwatch-summary>\n'
        return ret_str,totals,iters

global sw_timer
sw_timer = stopwatch()

