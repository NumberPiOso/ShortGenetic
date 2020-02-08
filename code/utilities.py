import sys
import os
import pandas as pd
from functools import wraps
from time import time


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def dict_to_pd_table(times):
    times_table = pd.DataFrame.from_dict(times, orient='index', columns=['time'])
    times_table['%'] = times_table['time']/ sum(times_table['time'])
    return times_table.round(2)

def read_file(fp):
    names = ['sta', 'x', 'y', 'dem', 'delete']
    table = pd.read_csv(fp, header=None, names=names, index_col=0,
                        usecols=names[:-1])
    return table

def track_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        elapsed_time_in_s = time() - start_time
        me = args[0]
        me._times[func.__name__] = (
            me._times.get(func.__name__, 0) + elapsed_time_in_s)
        return result
    return wrapper

