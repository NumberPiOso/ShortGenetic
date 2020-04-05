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



# Metrics used to compare frontiers

def get_metrics(frontiers_list):
    """
    Inputs:
        frontiers_list: (list [array])
            list of non dominated arrays of every model.
    
    Outputs:
        metrics (pandas dataframe):
            Ordered by frontier list with columns every metric 
            and rows are every frontier in frontier list.

        """
    # Get paretto aproximation frontier
    concat_frontiers = np.concatenate(frontiers_list)
    par_front = get_paretto(concat_frontiers)
    
    # Normalize frontiers
    limites = np.zeros([2, 2])
    limites[:, 0] ,limites[:, 1] = np.min(par_front, axis=0), np.max(par_front, axis=0)
    par_front = normalized(par_front, limites)

    # Get metrics to every frontier
    results = []
    for frontier in frontiers_list:
        n_frontier = normalized(frontier, limites)
        m1 = set_cov(par_front, n_frontier)
        m2 = gen_distance(n_frontier, par_front)
        m3 = spacing(n_frontier)
        m4 = eucl_sum(n_frontier)
        mm = [m1, m2, m3, m4]
        results.append(mm)

    return pd.DataFrame(results, columns=['set_cov', 'gen_dist', 'spacing', 'eucl_sum'])

def normalized(array, limites):
    new_array = np.zeros_like(array)
    for i in range(array.shape[1]):
        delta = limites[i,1] - limites[i,0]
        new_array[:, i] = (array[:, i] - limites[i, 0])/ delta
    return new_array
    

def get_paretto(array):
    cols = ['Unb', 'Tsp']
    df_frontiers = pd.DataFrame(array, columns=cols)
    df_frontiers.sort_values(cols, inplace=True)
    sorted_array = df_frontiers.values
    non_dom_pts = []
    l_tsp = np.inf
    for unb, tsp in sorted_array:
        if tsp < l_tsp:
            l_tsp = tsp
            non_dom_pts.append([unb, tsp])
    return np.array(non_dom_pts)


def set_cov(A, B):
    '''
    print(set_cov(n_frontier, par_front))
    debe ser 0 para cualquier frontera'''
    def weakly_dom(a, b):
        return (a[0] < b[0] and a[1] <= b[1]) or \
            (a[1] < b[1] and a[0] <= b[0])
    dominated_bs = 0
    for b in B:
        for a in A:
            if weakly_dom(a, b):
                dominated_bs += 1
                break
    return dominated_bs/ len(B)


def gen_distance(A, Pf, p=2):
    """
    print('---example gen distance ---')
    P = np.array([[1, 7.5],[1.1, 5],[2, 5],[3, 4],[4, 2.8],[5.5, 2.5],[6.8, 2.0],[8.4, 1.2]])
    A = np.array([[1.2, 7.8], [2.8, 5.1], [4, 2.8], [7, 2.2], [8.4, 1.2]])
    print(gen_distance(A, P))
    expected <-- .19"""
    def di(a,Pf):
        sum_squares = np.sum((Pf - a)**2, axis=1)
        return min(sum_squares) ** .5
    overall_dist = 0
    for a in A:
        overall_dist += di(a, Pf) ** p
    overall_dist = overall_dist ** (1/p)
    return  overall_dist/ len(A)


def spacing(A):
    '''Example:
        A = np.array([[1.2, 7.8], [2.8, 5.1], [4, 2.8], [7, 2.2], [8.4, 1.2]])
        print(spacing(A))
        expected_result <- .18
    '''
    if len(A) == 1:
        return 0

    def di_func(i, Q):
        counter = 0
        aux_d = np.zeros(len(Q)-1)
        for k in range(len(Q)):
            if(k == i):
                continue
            aux_d[counter] = sum(np.abs(A[i] - A[k]))
            counter += 1
        return min(aux_d)
    n = len(A)
    d_vec = np.zeros(n)
    for i in range(n):
        d_vec[i] = di_func(i, A)
    return np.std(d_vec, ddof=0)


def eucl_sum(A):
    array_subs = (A[1:] - A[:-1]) **2
    distances = np.sum(array_subs, axis=1) **.5
    return sum(distances)