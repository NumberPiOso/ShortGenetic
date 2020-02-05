import utilities as util
from model import Sol, SolCollection, read_stations
# import model as m
import numpy as np
import pandas as pd
import random
np.random.seed(15154)
random.seed(123347)

# Read file and stations
fp = '~/Dropbox/PI/PI2/data/n5q10A.dat'
data = util.read_file(fp)
stations = read_stations(data) # list of stations
Sol.set_stations(stations)

# Create random solutions
k = len(stations)
order = [ *range(k)]
random.shuffle(order)
sol1 = Sol(order)
sol1.display()

