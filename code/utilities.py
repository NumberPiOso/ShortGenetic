import sys
import os
import pandas as pd

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
