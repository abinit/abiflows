from __future__ import print_function, division, unicode_literals, absolute_import

import prettytable as pt

from pymatgen.util.serialization import pmg_serialize
from monty.json import MSONable


def seconds_to_hms(seconds):
    """
    Converts second to the format "h:mm:ss"

    Args:
        seconds: number of seconds

    Returns:
        A string representing the seconds with the format "h:mm:ss". An empty string if seconds is None.
    """
    if seconds is None:
        return ""

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


class TimeReport(MSONable):
    """
    Report of the time consumed for the whole workflow and for each single step.
    Includes both the time required for the calculations as well as the cpu time consumed.
    """

    def __init__(self, total_run_time, n_fws, total_cpu_time=None, contributed_cpu_time=0, total_run_time_per_tag=None,
                 total_cpu_time_per_tag=None, contributed_cpu_time_per_tag=None, worker=None):
        self.total_run_time = total_run_time
        self.n_fws = n_fws
        self.total_cpu_time = total_cpu_time
        self.contributed_cpu_time = contributed_cpu_time
        self.total_run_time_per_tag = total_run_time_per_tag
        self.total_cpu_time_per_tag = total_cpu_time_per_tag
        self.contributed_cpu_time_per_tag = contributed_cpu_time_per_tag
        self.worker = worker

    @pmg_serialize
    def as_dict(self):
        d = dict(total_run_time=self.total_run_time, n_fws=self.n_fws, total_cpu_time=self.total_cpu_time,
                 contributed_cpu_time=self.contributed_cpu_time, total_run_time_per_tag=self.total_run_time_per_tag,
                 total_cpu_time_per_tag=self.total_cpu_time_per_tag, contributed_cpu_time_per_tag=self.contributed_cpu_time_per_tag,
                 worker=self.worker)

        return d

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d.pop("@module", None)
        d.pop("@class", None)
        return cls(**d)

    def __str__(self):
        s = ''
        if self.worker:
            s += 'Worker: {}\n'.format(self.worker)

        s += 'Total number of fireworks: {}\n\n'.format(self.n_fws)

        t = pt.PrettyTable(['task tag', 'run time (h:m:s)', 'cpu time (h:m:s)', 'N cpu time'], float_format="5.3")
        t.align['task tag'] = 'l'
        if self.total_run_time_per_tag:
            for task_tag in self.total_run_time_per_tag.keys():
                # t.add_row([task_tag, self.total_run_time_per_tag[task_tag]/3600, self.total_cpu_time_per_tag.get(task_tag, 0)/3600,
                #           self.contributed_cpu_time_per_tag.get(task_tag, 0)])
                t.add_row([task_tag, seconds_to_hms(self.total_run_time_per_tag[task_tag]), seconds_to_hms(self.total_cpu_time_per_tag.get(task_tag, 0)),
                          self.contributed_cpu_time_per_tag.get(task_tag, 0)])

        # t.add_row(['Total', self.total_run_time/3600, self.total_cpu_time/3600, self.contributed_cpu_time])
        t.add_row(['Total', seconds_to_hms(self.total_run_time), seconds_to_hms(self.total_cpu_time), self.contributed_cpu_time])

        s += str(t)

        return s
