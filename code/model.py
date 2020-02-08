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
        route: list[ints]
            The order of the stations
        cost_unb: float or int
            The cost function associated with unbalance in BRP.
        cost_tsp: float
            The cost of the route (calculated with distance between stations)
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
    

    def reproduce(self, other, interactive=False):
        '''
        Input:
            self: Object type sol
            other: Object type sol
        Output:
            new_sol: Object type sol
                new_sol is the first son of self and other done
                by method described in page 7 in 
                simple and efective evolutionary algorithm for the vehicle 
                routing problem by Christian Prins (2004).
        '''
        # Parameters
        route1 = self.route
        route2 = other.route
        n = len(route1)
        
        # Define starting and ending point
        start, end = sorted( random.sample( range(0,n), 2))
        i, j = start, end+1 # For being end inclusive
        
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
        # print(new_route)
        end1, end2 = self.index_end_route, other.index_end_route
        end_new_route = random.sample([end1, end2], k=1)[0] # def returns list
        new_sol = Sol(new_route, end_new_route)
        if interactive:
            print(f'Cut points are {i, j}')
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
        successors = set( [(route[i], route[i+1]) for i in range(n-1)])
        return successors

    def one_gen_mut(self):
        n = len(self.route)
        # i and j can be the same
        i, j = np.random.randint(0, n, 2)
        self.route[i], self.route[j] = self.route[j], self.route[i]
        self.index_end_route = np.random.randint(2, n)

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
        self.n_elite = n_pob// 2
        self.n_random_sols = num_random_sols
        self.n_mutations = int(round(ratio_mutation * self.k ))

        # Best sols info
        self._times = {} # Track times, done by decorator @track_time 
        self.best_sol = self.sols[0]
        self.best_sol_value_unb = np.inf
        self.best_sol_value_tsp = np.inf
        self.list_points = []

    def train_time(self, max_time, interactive=False, n_chks=4 ,ret_times=False):
        'Time is in seconds'
        time_to_chunk = 0
        chuncked_times, chuncked_iters  = [], [] # To return plots in interactive, 
        overall_time = 0
        # times = dict()
        levels_table = self.medal_table()
        iters = 0
        while overall_time < max_time:
            iters += 1
            time_init = time()
            levels_table = self.medal_table(interactive)
            # times =  self.time_it("Medals", times)
            parents = self.parent_selection()
            # times =  self.time_it("parents", times)
            sols = self.reproduce(parents)
            # times =  self.time_it("crossover", times)
            self.one_gen_mutation()
            # times =  self.time_it("mutation", times)
            overall_time += time() - time_init
            if time_to_chunk < overall_time:
                time_to_chunk += max_time /n_chks
                chuncked_times.append(overall_time)
                chuncked_iters.append(iters)
                if interactive:
                    bests = levels_table[levels_table['Pod_lev'] == 0]
                    self.list_points.append(bests.values[:, 0:2]) #save results
                    print('---------------------------------------------------------')
                    print(f'time := {overall_time:.2f}.  iterations : {iters}')
                    print(parents)
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
        self.sols = [self.sols[ii] for ii in order_solutions[:self.k]]
        return table

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
    def reproduce(self, parents):
        sons = []
        for (j1, j2) in parents:
            son = self.sols[j1].reproduce(self.sols[j2])
            sons.append(son)
        self.sols = [*self.sols, *sons]
        return self.sols

    def local_searchs(self, levels_table, removed_sols):
        '''Some basic steps
        1. Get solutions with costs
        2. Normalize
        3. psols <- Non dominated non_repeated sols
        4. a <- Solution with biggest vicinity
        5. w*, z* <- Optimize model(table, a)
        6. Get z_i for all solutions.
        6.1 take out z_i that come from trivial solution
        7. Select sols with more number of z_i below z_i + e
           (A possible solution for e coul be the other extreme of a)
        8. Local search selected sols.
        '''
        # 1. Get all solutions with costs
        def delete_indexes_deleted(table, removed_sols):
            '''Inputs are the table before the solutions remotions and
            the indices of the removed sols. It's output its the table
            with the indices of the solutions and withouth any of the 
            corrected sols.
            Example:
            levels table ->
            index  Unb  Tsp  Sol
            2   6   520 0
            8   6   520 1
            28   6   520 5
            33   6   520 6
            37   6   520 7
            43   6   520 8
            47   6   520 9
            21  12  252 4
            12  20  198 2
            16  20  198 3
            
            removed_sols -> [1,2,4,5,6]

            Output:
                        index  Unb  Tsp  Sol
                2   6   520 0
                37   6   520 2
                43   6   520 3
                47   6   520 4
                16  20  198 1
            '''
            pd.options.mode.chained_assignment = None  # default='warn'
            l = removed_sols
            assert(all(l[i] <= l[i+1] for i in range(len(l)-1))) # is sorted    
            # Delete
            for i in removed_sols:
                table = table[table['Sol'] != i]
            # print(len(table['Sol'].unique()))
            for i in reversed(removed_sols):    
                table.loc[table['Sol'] > i,'Sol'] -= 1
            return table

        table = delete_indexes_deleted(levels_table, removed_sols)
        # 2. Normalize
        norm_cols = ['Unb', 'Tsp']
        df = table[norm_cols]
        df =(df - df.min())/ (df.max() - df.min())
        table[norm_cols] = df
        # 3. psols <- Non dominated non_repeated sols
        psols = table[table['Pod_lev'] == 0]
        # print('-----------------------------')
        # # print(table)
        # print(psols)
        psols.drop_duplicates(subset=['Unb', 'Tsp'], inplace=True)
        if len(psols) <= 2:
            # print('Local search under this model could not be implemented')
            # print('Only two pareto sols,what a shame')
            return self.sols
        # 4. a <- Solution with biggest vicinity
        values = deepcopy(psols[['Unb', 'Tsp']].values)     
        distances =  np.sqrt(np.sum((values[:-1] - values[1:])**2, axis=1))
            # distance i to i+1
        vicinities = distances[:-1] + distances[1:]
        a = 1 + np.argmax(vicinities) # First do not have vicinity
        # print(' Non dominated sols ')
        # print(values)
        # print('distances')
        # print(distances)
        # print('Biggest vicinity ', a)
        # 5. w*, z* <- Optimize model(table, a)
        ba, ca = values[a]
        bant, cant =  values[a-1]
        cota_sup = (bant - ba)/ (ca - cant + bant - ba)
        bsig, csig = values[a+1]
        cota_inf = (ba - bsig)/ (csig - ca + ba - bsig)
        if ca - ba > 0:
            wopt = cota_sup
            zopt = wopt * (ca - ba) + ba
            # epsilon = cota_inf * (ca - ba) + ba - zopt 
            # z_max = cota_inf * (ca - ba) + ba
            # z_max = wopt * (csig - bsig) + bsig
            # z_max = wopt * (csig - bsig) + ba
            z_max = 1.1 * zopt
        elif ca - ba < 0:
            # print(ca - ba)
            wopt = cota_inf
            zopt = wopt * (ca - ba) + ba
            # z_max = cota_sup * (ca - ba) + ba
            # z_max = wopt * (cant - bant) + bant
            # z_max = wopt * (cant - bant) + ba
            z_max = 1.1 * zopt
        else:
            raise ValueError(' el modelo de optimizacion no tiene sentido')
        # epsilon = .1*zopt
        # assert( epsilon >= 0)
        assert( z_max > zopt )
        # 6. Get z_i for all solutions.
        z_func = lambda b, c: wopt * (c-b) + b
        table['Zi'] = z_func(table['Unb'], table['Tsp'])
        # 6.1 delete trivials
        table = table[table['Tsp'] != 0]
        # 7. select solutios
        selected = table['Sol'][table['Zi'] < z_max]
        selected = selected.unique()
        # print(' From ', len(table['Sol'].unique()), 'selections possible,s\
        #      Just selected' , len(selected))
        # print('Selected solutions ', selected)
        # 8. LS selected sols
        all_lss = []
        #local search for repeated is O(1) and returns empty
        for s in list(selected):
            ls_sols = self.sols[s].ls_fast2opt( self.rt_non_dominance)
            if ls_sols:
                all_lss.append(ls_sols)
        new_sols = [item for sublist in all_lss for item in sublist]
        if new_sols:
            self.sols = [*self.sols, *new_sols]

        #plt.plot(table['Tsp'], table['Unb'])
        # Tournament local search
        #     list_sols = range(len(self.sols))
        #     i = min(random.sample(list_sols, 2))
        #     if i not in already_used:
        #         new_sols = self.sols[i].ls_fast2opt( self.rt_non_dominance)
        #         if new_sols: # it could be empty if local optimal
        #             # print('local search suceed.')
        #             self.sols = [*self.sols, *new_sols]
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
        if num_sols < sols_that_stay:
            r = sols_that_stay - num_sols
            new_sols = self.gen_random_sols(r)
            self.sols = [*self.sols, *new_sols]
        random_sols = self.gen_random_sols(self.n_random_sols)
        self.sols = [*self.sols[:sols_that_stay], *random_sols]
        return self.sols

    def plot_chunks(self, label='', max_time=10):
        k1 = len(self.list_points)
        # alpha = np.linspace(.3, 1, num=k1)
        colors = ['r', 'b', 'g', 'c', 'm']
        for ri, points in enumerate(self.list_points):
            ##### POINTS CLEANUP, LOOKING FOR NON DOMINATED #####
            org_criteria = ['Unb', 'Tsp']
            table = pd.DataFrame(points,columns=org_criteria)
            table.sort_values(org_criteria, inplace=True)
            non_dom_pts = []
            l_dem, l_tsp = table.iloc[0]
            non_dom_pts.append([l_dem, l_tsp])
            for i, row in enumerate(table.itertuples()):
                __, dem, tsp = row
                if i == 0:
                    non_dom_pts.append([dem, tsp])
                else:
                    if tsp < l_tsp:
                        l_dem, l_tsp = dem, tsp
                        non_dom_pts.append([l_dem, l_tsp])
            non_dom_pts = np.array(non_dom_pts)
            ##### POINTS CLEANUP, LOOKING FOR NON DOMINATED #####
            # plt.plot(non_dom_pts[:,1], non_dom_pts[:,0], c=colors[ri], label=f'chunck {ri}')
            plt.scatter(non_dom_pts[:,1], non_dom_pts[:,0], c=colors[ri], label=f'time {max_time*(ri+1)// 4}')
        plt.xlabel('Route cost')
        plt.ylabel('Unbalance cost')
        plt.title('Pareto Curve for different chunks')
        plt.legend()
        # if not label:
        #     label = f'_iterations{self.n_iters}'
        # plt.savefig(f'./results/evolution_frontier/n_pob{self.k}{label}.png')
        return plt

    def gen_random_sols(self, num_sols=0):
        n = len(self.sols[0])
        new_rks = np.random.random((num_sols, n))
        new_sols = [Sol(rks) for rks in new_rks]
        return new_sols

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

