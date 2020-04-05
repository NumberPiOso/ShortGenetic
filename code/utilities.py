import sys
import os
import pandas as pd
import numpy as np
from functools import wraps
from time import time
import matplotlib.pyplot as plt


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
        me._times[func.__name__] +=  elapsed_time_in_s
        return result
    return wrapper


def get_lims_pts(list_points):
    get_lim = lambda func, axis: func([func(matrix[:, axis]) for matrix in list_points])
    x_lims = (get_lim(min, 0), get_lim(max, 0))
    y_lims = (get_lim(min, 1), get_lim(max, 1))
    return(x_lims, y_lims)

def normalize_list_in_place(list_points, get_lims_pts=get_lims_pts):
    '''
        Normalization in place. Normalization of form (x - x_min)/ (x_max - x_min)
        Inputs:
            list_points: list of arrays
                Every array represents points [x, y] by its columns.
        No outputs:
            Normalization is done in place
        
    '''
    
    def normalize_in_pl(array, x_lim, y_lim):
            min_x, max_x = x_lims
            array[:, 0] -= min_x
            array[:, 0] /= (max_x - min_x)
            
            min_y, max_y = y_lims
            array[:, 1] -= min_y
            array[:, 1] /= (max_y - min_y)
            
    x_lims, y_lims = get_lims_pts(list_points)
    for array in list_points:
        normalize_in_pl(array, x_lims, y_lims)

def plot_list_points(list_points, iters_list, times_list, n=0):
    points = list_points[n]
    x, y = points[:,0], points[:,1]
    plt.figure(figsize=(14, 6))
    plt.scatter(x, y)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title(f'time := {times_list[n]:.2f}.  iterations : {iters_list[n]}')
    plt.xlabel('Unb')
    plt.ylabel('Tsp')
    plt.grid()
    plt.show()
    return plt


### New random solutions
def closest_station(stat_i, sel_stats, sel_stats_ind):
    distances = [stat_i.distance(stat_j)  for stat_j in sel_stats]
    argmin_dist = np.argmin(distances)
    return sel_stats_ind[argmin_dist]