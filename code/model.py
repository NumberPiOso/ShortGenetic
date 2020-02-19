import pandas as pd
import numpy as np
from utilities import *
import random
from copy import deepcopy
from operator import itemgetter
import matplotlib.pyplot as plt
from time import time
import sys, os
from level_function import  cy_medal_organizer
from collections import defaultdict

def read_stations(table):
    stations = []
    for tup in table.iterrows():
        line = tup[1]
        x = line['x']
        y = line['y']
        dem = line['dem']
        stat = Station(x, y, dem)
        stations.append(stat)
    Sol.set_stations(stations)
    return stations

def normalized_order(order):
    '''after taking out the alike comparisons, the order of the solutions
            may refer to solutions index that are not availaible now, so a 
            normalization to the indexes must be applied O (n^2) 
            examples:
            input     ---> output
            [3, 7, 6] --->  [0 2 1]
            [3, 6, 7] --->  [0 1 2]
            [7, 4, 1] --->  [2 1 0]
            [7, 4, 1, 5] --->  [3 1 0 2]'''
    order = np.array(order)
    new_order = np.zeros(len(order), dtype=int)
    for i, el in enumerate(order):
        new_order[i] = np.sum([order < el])
    return new_order

def get_paretto_from_array(array):
    columns = ['Unb', 'Tsp']
    table = pd.DataFrame(array, columns=columns)
    table.sort_values(columns, inplace=True)
    org_costs = table.values
    paretto_sols = []
    low_tsp = np.inf
    for unb, tsp in org_costs:
        if tsp < low_tsp:
            low_tsp = tsp
            paretto_sols.append([unb, tsp])
    return np.array(paretto_sols)


class Sol:
    """
    A class used to represent a solution tp BRP.

    Attributes:
        _stations: class attribute list(station)
            Represents all stations of the problem.
        route: list[ints]
            The order of the stations
        cost_unb: float or int
            The cost function associated with unbalance in BRP.
        cost_tsp: float
            The cost of the route (calculated with distance between stations)
        _succesors: set(int, int)
            The arcs present at every solution
    Methods: 
        calc_cost_tsp
        calc_cost_unbalance
        reproduce(self, other):
            Reproduces two solutions with one cross to produce a child.
        two_opt_sol
    """
    _stations = []
    def __init__(self, route, index_end_route=None):
        if not Sol._stations:
            raise Exception("""Please initialize _stations before defining 
                solutions. Use method Sol.set_stations for this task""")
        if index_end_route is not None:
            self.index_end_route = index_end_route
        else:
           self.index_end_route = len(route) 
        # Check if every item is from type int
        # assert(all(np.issubdtype(item, np.integer) for item in route)) 
        self.route = route
        # Where the route ends (exclusive)
        self.cost_unb, self._solution = self.calc_cost_unbalance()
        self.cost_tsp = self.calc_cost_tsp(self._solution)
        # Successors set
        self._successors = self.successors_set()

    @classmethod
    def random_sol(cls, end=None):
        k = len(Sol._stations)
        order = [ *range(k)]
        random.shuffle(order)
        if end is None:
            end=k
        return(cls(order, end))

    def calc_cost_unbalance(self):
        '''
            Calculates inbalance of the route
            Inputs(implicit):
                Sol._stations: (set of stations)
                self.route
            Output:
                cost_unbalance
        '''
        route = self.route
        demands = [Sol._stations[i].dem for i in route]
        end_r = self.index_end_route
        # Solution is bikes mounted at every station
        solution = constructivo1(demands[:end_r])
        # First sumand
        norm1 = lambda l: sum(np.abs(l))
        cost_unbalance = norm1(solution + demands[:end_r]) \
                        + norm1(demands[end_r:]) 
        return cost_unbalance, solution

    def calc_cost_tsp(self, solucion):
        '''
            Input: 
                solution:  (list ints : len(route[:end_route]))
                The number of bikes mounted at every station.
            Output:
            cost_route: The cost of the full route without passing by the 
                stations that solution does not have to leave or pick up bikes. 
        '''
        route = self.route
        stations = Sol._stations
        # The stater index (not starting in stations that do not need bikes)
        r_ini = next((i for i, x in enumerate(solucion) if x), 0)
        stat_i = stations[route[r_ini]]
        cost = 0
        for r in range(r_ini+1, self.index_end_route):
            # if solucion[r] is 0 I do not need to go there
            if solucion[r]:
                j = route[r]
                stat_j = stations[j]
                cost += stat_i.distance(stat_j)
                stat_i = stat_j
        return cost

    def reproduce(self, other, option=1, interactive=False):
        '''
        Input:
            self: Object type sol
            other: Object type sol
            option (int):
                1 -> new_sol is the first son of self and other done
                by method described in page 7 in 
                simple and efective evolutionary algorithm for the vehicle 
                routing problem by Christian Prins (2004).
                2 -> Solution is getting the subroute i, j of the second route son
                     and trying to accomodate it in the first route so it
                     is TSP cost optimal
        Output:
            new_sol: Object type sol
                
        '''
        # Parameters
        route1 = self.route
        route2 = other.route
        n = len(route1)
        
        # Define starting and ending point
        start, end = sorted( random.sample( range(0,n-1),2))
        i, j = start, end+1 # For being end inclusive for option1
        if option == 1:
            new_route = np.ones(n, dtype=int)*(-1)

            # The middle ones are the same
            new_route[i:j] = route1[i:j]

            i2 = j
            for i1 in range(j, n+i):
                while route2[i2%n] in new_route:
                    i2 += 1
                    assert(i2 < n+i+j)
                new_route[i1%n] = route2[i2%n]
                i2 += 1
        elif option == 2:            
            subi_j = route2[i:j]
            other_nodes = [x for x in route1 if x not in subi_j]
            # The subroute subi_j is going to be placed at every arc of
            # other nodes. With savings algorithm, find the best.
            first_station_sub = Sol._stations[subi_j[0]]
            last_station_sub = Sol._stations[subi_j[-1]]
            best_index = -1
            best_saving = -np.inf
            for k in range(len(other_nodes)+1):
                if k == 0:
                    i = other_nodes[0]
                    stat_i = Sol._stations[i]
                    saving = -last_station_sub.distance(stat_i)
                    # first is best
                elif k == len(other_nodes):
                    # Check this bound
                    saving = - stat_i.distance(first_station_sub)
                else:
                    stat_i_1 = stat_i
                    i = other_nodes[k]
                    stat_i = Sol._stations[i]
                    saving = stat_i.distance(stat_i_1) \
                            - stat_i_1.distance(first_station_sub) \
                            - last_station_sub.distance(stat_i) 
                if saving > best_saving:
                    best_saving = saving
                    best_index = k
                if interactive:
                    print(f" Ahorro en {k-1} -> {k}: {saving} ")
            bi = best_index
            new_route = [*other_nodes[:bi], *subi_j, *other_nodes[bi:]]
        # At this point I already have defined the new_route (int) vector.
        
        end1, end2 = self.index_end_route, other.index_end_route
        end_new_route = random.sample([end1, end2], k=1)[0] # sample returns list
        new_sol = Sol(new_route, end_new_route)
        if interactive:
            print(f'Cut points are {start, end+1}')
            print(f'End route has chosen to be {end_new_route}')
            new_sol.display()
        return new_sol

    @classmethod
    def two_opt_sol(cls, rks, i, j, route):
        # Creates a Sol object resulted from applying 2opt with indexes (i,j) 
        new_rks = deepcopy(rks)
        num_changes = (j-i)//2
        for change in range(num_changes):
            f1, f2 = route[(i+1)+change], route[j-change]     
            new_rks[f1], new_rks[f2] = new_rks[f2], new_rks[f1]
        return cls(new_rks)

    @classmethod
    def set_stations(cls, stations):
        cls._stations = stations

    def sub_sols(self):
        route = self.route
        subsols = []
        for i in range(1,len(route)+1):
            subsols.append(route[:i])
        return subsols

    def successors_set(self):
        route = self.route
        n = len(route)
        successors = set(((route[i], route[i+1]) for i in range(n-1)))
        return successors

    def isLike(self, other):
        n = len(self._successors)
        succ_1 = self._successors
        succ_2 = other._successors
        likeness = len(succ_1.intersection(succ_2)) / (n-1)
        return likeness

    def one_gen_mut(self):
        n = len(self.route)
        # i and j can be the same
        i, j = np.random.randint(0, n, 2)
        self.route[i], self.route[j] = self.route[j], self.route[i]
        # self.index_end_route = np.random.randint(2, n)

    def display(self):
        print(f'''---------Solution object---------
                route: {self.route}
                Ends at node #: {self.index_end_route}
                The constructive sol is: {self._solution}
                Tsp cost: {self.cost_tsp}
                Unbalance costs: {self.cost_unb}''')

    def __len__(self):
        return len(self.route[:self.index_end_route])

    def __str__(self):
        return f'route: {self.route},\
            end route at_ {self.route[self.index_end_route-1]} '

    def __repr__(self):
        return f'route: {self.route[:self.index_end_route]}\n'


class SolCollection:
    """
    A collection of all solutions
    ---
    Attributes:
        sols: (list(Sol))
            The list of solutions
        k: (int)
            Refers to the number of individuals type Sol in the list sols.
        n_sons: (int) 
            Number of sons to have at every iteration.
        n_random_sols: (int)
            Number of random solutions to include at every iteration.
        
    """
    def __init__(self, n_pob, ratio_sons=.6,
                ratio_mutation = 0.2, num_random_sols=0):
        # Define solutions
        self.sols = [Sol.random_sol() for ii in range(n_pob)]
        self.k = n_pob

        # Parameters of model
        self.n_sons = int(round(ratio_sons * self.k))
        self.n_elite = n_pob
        self.n_random_sols = num_random_sols
        self.n_mutations = int(round(ratio_mutation * self.k ))
         # If two solutions are equal in at least 80% arcs. Do not show mercy.
        self.max_like = .80

        # Best sols info
        self._times = defaultdict(int) # Track times, done by decorator @track_time 
        self.best_sol = self.sols[0]
        self.best_sol_value_unb = np.inf
        self.best_sol_value_tsp = np.inf
        self.list_points = []

    def train_time(self, max_time, interactive=False, n_chks=4 ,ret_times=False,reproduction_type=1, print_=False):
        'Time is in seconds'
        time_to_chunk = 0
        chuncked_times, chuncked_iters  = [], [] # To return plots in interactive, 
        overall_time = 0
        best_points = [[1, 1]]
        # times = dict()
        levels_table = self.medal_table()
        iters = 0
        while overall_time < max_time:
            iters += 1
            time_init = time()
            # Medal table,  take out alikes and replace poblation
            levels_table = self.medal_table(interactive)
            removed_sols = self.take_out_alikes()
            sols = self.poblation_replacement(include_randoms=True)
            # Parent Selection
            parents = self.parent_selection()
            # Reproduction
            sols = self.reproduce(parents, option=reproduction_type)
            # Mutation
            self.one_gen_mutation()
            overall_time += time() - time_init
            if time_to_chunk < overall_time:
                time_to_chunk += max_time /n_chks
                chuncked_times.append(overall_time)
                chuncked_iters.append(iters)
                if interactive:
                    bests = levels_table[levels_table['Pod_lev'] == 0].values[:,0:2]
                    self.list_points.append(bests) #save results
                if print_:
                    print('---------------------------------------------------------')
                    print(f'time := {overall_time:.2f}.  iterations : {iters}')
                    # print(parents)
                    print(sols)
        if interactive:
            return(self.list_points, chuncked_times, chuncked_iters)
        # if ret_times:        
        #     return dict_to_pd_table(times)
        return self.get_paretto_sols()

    def get_paretto_sols(self):
        n = len(self.sols[0])
        k = len(self.sols)
        # Get all costs
        costs = np.zeros([k*n, 2])
        for i,sol in enumerate(self.sols):
            costs_unb = sol.cost_unb
            costs_tsp = sol.cost_tsp
            costs[i*n: (i+1)*n] = np.transpose([costs_unb, costs_tsp])
        
        # organize by columns
        columns = ['Unb', 'Tsp']
        table = pd.DataFrame(costs, columns=columns)
        table.sort_values(columns, inplace=True) 
        org_costs = table.values
        paretto_sols = []
        low_tsp = np.inf
        for unb, tsp in org_costs:
            if tsp < low_tsp:
                low_tsp = tsp
                paretto_sols.append([unb, tsp])
        return np.array(paretto_sols)
    
    def get_all_costs(self, named=False):
        '''If named False:
            returns a matrix of cost with following template
            [ b_i, t_i] for every i in all sols and subsols.
            
            if named True:
                return a pandas dataframe'''
        n = len(self.sols[0])
        k = len(self.sols)
        costs = np.zeros([k*n, 2])
        for i,sol in enumerate(self.sols):
            costs_unb = sol.cost_unb
            costs_tsp = sol.cost_tsp
            costs[i*n: (i+1)*n] = np.transpose([costs_unb, costs_tsp])
        return costs

    @track_time
    def medal_table(self, show=False):
        '''Find the non dominance level that a solution is in. 
        Then sorts the solutions according to that level, lower -> better.
        '''
        # Get costs
        columns = ['Unb', 'Tsp', 'Sol', 'Pod_lev']
        data = []
        for i, sol in enumerate(self.sols):
            unb = sol.cost_unb
            tsp = sol.cost_tsp
            data.append([unb, tsp, i, -1])
        table = pd.DataFrame(data, columns=columns)
        table.index.name = 'Solution'
        n = len(self.sols)
        
        # Sort table
        org_criteria = ['Unb', 'Tsp']
        table.sort_values(org_criteria, inplace=True)
        d = table.values[:, 0]
        c = table.values[:, 1]
        l = np.repeat(-1, len(c))
        cy_medal_organizer(d, c, l, n)
        table['Pod_lev'] = l

        # What to do with solutions which TSP == 0
        # table['Pod_lev'][table['Tsp' == 0]] = 3
        # Sort solutions from best to worst
        table.sort_values(['Pod_lev'], inplace=True)
        order_solutions = table.index
        self.sols = [self.sols[ii] for ii in order_solutions]
        return table

    @track_time
    def take_out_alikes(self):
        orig_num = len(self.sols)
        rem_sols_index = []
        act_num_sols = orig_num
        cont = 0
        i = 0
        while (i < act_num_sols-1):
            sol_i = self.sols[i]
            sol_j = self.sols[i+1]
            # print('Take out alikes method')
            # print(sol_i)
            # print(sol_j)
            # print('ratio :', sol_i.isLike(sol_j))
            if (sol_i.isLike(sol_j) > self.max_like):
                self.sols.pop(i+1)
                rem_sols_index.append( i+1+(orig_num-act_num_sol))
                act_num_sol -= 1
            else:
                i += 1
        # if orig_num - act_num_sol:
        #     print(f'Se quitaron {orig_num - act_num_sol} soluciones por parecidas.')
        #     print('Las de indices ', rem_sols_index)
        return rem_sols_index

    @track_time
    def parent_selection(self):
        ''' Function to select parents from self.sols to do the apareation '''
        n_sons = self.n_sons
        parents = np.empty([n_sons, 2], dtype=int)
        # random.seed(9)
        for i1 in range(n_sons):
            parents[i1] = self.biased_parents()
        self.parents = parents
        return parents

    @track_time
    def biased_parents(self):
        '''
        Considers that solutions are organized from the best to the worst.
        returns a single couple with bias for good solutions.'''
        biassed_parents = range(self.n_elite)
        poss_parents = range(len(self.sols))
        i, j = random.sample(biassed_parents, 2)
        parent1 = min(i, j)
        while True:
            i, j = random.sample(poss_parents, 2)
            parent2 = min(i, j)
            if parent2 != parent1:
                break
        return [parent1, parent2]

    @track_time
    def reproduce(self, parents, option=1):
        '''
        Reproduce the solutions of the parents using method Sol.reproduce.
        Inputs:
            parents (list[int, int])
                List of indexes of the parents.
            option (int):
                Option to pass directly to reproduce method.
        Outputs:
            self.sols: list(sols)
                A list with the original population concatenated with the sons
        '''
        sons = []
        for (j1, j2) in parents:
            son = self.sols[j1].reproduce(self.sols[j2])
            sons.append(son)
        self.sols = [*self.sols, *sons]
        return self.sols


    @track_time
    def one_gen_mutation(self):
        k = len(self.sols)
        n_mutations = self.n_mutations
        for __ in range(n_mutations):
            index_sol = np.random.randint(0, k)
            self.sols[index_sol].one_gen_mut()

    def poblation_replacement(self):
        '''Keep the best solutions, but change the last ones with random ones.'''
        num_sols = len(self.sols)
        sols_that_stay = self.k - self.n_random_sols
        # Special case where I deleted too many sols.
        if num_sols < sols_that_stay:
            rn = sols_that_stay - num_sols
            new_sols = [Sol.random_sol() for __ in range(rn)]
            self.sols = [*self.sols, *new_sols]
        # General case
        random_sols = [Sol.random_sol() for __ in range(self.n_random_sols)]
        self.sols = [*random_sols, *self.sols[:sols_that_stay]] # By this order, random are given advantage
        return self.sols


    def get_times(self): 
        data = self._times
        df_times = pd.DataFrame.from_dict(data, orient='index', columns=['time(s)'])
        df_times['%'] = df_times['time(s)']/ sum(df_times['time(s)']) 
        return df_times
  
    def __repr__(self):
        return (str(self.sols))


class Station:
    def __init__(self, x, y, dem):
        self.x = x
        self.y = y
        self.dem = dem

    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** .5

# Constructivo 1

def inter_camion_station(actual, capacidad, demanda): # Esto es para el constructivo 1
    """inputs:
            camion contiene [numero bicicletas actuales cargadas,capacidad]
            demanda de la estacion
    outputs:
            camion que cambia sus bicicletas actuales
            y la nueva demanda de la estación"""
    if (demanda == 0):
        return actual, demanda
    
    # else
    montadas_camion = - min(actual, demanda) # Bicicletas a cargar al camion

    # Si el num de bicicletas a montar sobrepasaria la capacidad
    if (actual + montadas_camion) > capacidad:
        montadas_camion  = capacidad - actual # Monto el maximo

    actual = actual + montadas_camion
    demanda_new = demanda +  montadas_camion
    return actual, demanda_new

def constructivo1(demandas, capacidad=10):
    '''Constructivo usual, recojo o dejo el numero de bicicletas que me pidan... si puedo'''
    k = len(demandas)
    bikes_montadas = np.zeros(k, dtype=int)
    actual = 0
    for i,demanda in enumerate(demandas):
        # Defino cual es la estacion que debo visitar
        bikes_camion = actual

        # Veo si es la ultima estacion, cuyo caso dejo todo
        if (i == k-1): 
            # demanda_act = demanda - actual
            actual = 0
            bikes_montadas[i] = -bikes_camion
            break

        actual,demanda_act = inter_camion_station(actual, capacidad, demanda)
        # Guardo la demanda restante para ser totalizada en la funcion de costo
        bikes_montadas[i] = actual - bikes_camion
    return bikes_montadas

