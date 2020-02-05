import pandas as pd
import numpy as np
from model import *
import random
import glob
import os
from time import time
np.random.seed(15154)
random.seed(125445)



fp = '~/Dropbox/PI/PI2/data/n50q10A.dat'
data = read_file(fp)
stations = read_stations(data) # list of stations
Sol.set_stations(stations)
k = len(stations)
n_pob = 40


# other_solutions = SolCollection(pob, stations, ratio_sons=1, ratio_elites=.6,
#                             ratio_mutation = 0.2, max_like=.0, num_random_sols=0,
#                              rt_non_dominance=0.6)
# times2 = other_solutions.train_time(25, show=False)
# print(times2)
# enablePrint()
# print('For n_iter,  n_pob')
#         print(times)


elapsed_time = 0
i5 = 1000
for i in range(i5):
    pob = np.random.random((n_pob, k))
    # # Starts iteration
    solutions = SolCollection(pob, stations, ratio_sons=1, ratio_elites=.6,
                                ratio_mutation = 0.2, max_like=.90, num_random_sols=0,
                                rt_non_dominance=0.6)
    # times = solutions.train_time(1, ret_times=True, plot_advances="chuncks")
    # print(times)





    # show_chunk = True

    time_init = time()
    levels_table, medal_table = solutions.medal_table()
    elapsed_time += time() - time_init
    # print(levels_table)
    # parents = solutions.parent_selection()
    # sols = solutions.gen_crossover(parents)
    # solutions.one_gen_mutation()
    # levels_table, medal_table = solutions.medal_table()
    # print(levels_table)
    # removed_sols = solutions.take_out_alikes()
    # print('---------------------------')
    # solutions.local_searchs(levels_table, removed_sols)
    # levels_table, medal_table = solutions.medal_table(show_chunk)
    # sols = solutions.poblation_replacement()
print('Promedio por tabla :', elapsed_time/ i5)