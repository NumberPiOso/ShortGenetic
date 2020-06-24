import pandas as pd
import numpy as np
import random
import os
import sys
import re
from copy import deepcopy
sys.path.insert(1, '..')
from model import SolCollection, Sol,read_stations
import utilities as util
from utilities import get_paretto, set_cov, gen_distance, spacing, eucl_sum, normalized

def ls():
    names = os.listdir()
    names.sort()
    return names

def array_from_txt(cwd_path):
    array = np.genfromtxt(cwd_path)
    if array.ndim == 1:
        array = np.array([array])
    return array

def create_folder(name):
    try:{
        os.mkdir(name)
    }
    except Exception as e:
        pass

def read_frontiers_infolder(cwd_path, dire):
    os.chdir(cwd_path + '/' + dire)
    files = ls()
    params = []
    frontiers = []
    for file_name in files:
        if file_name.endswith('.txt'):
            params.append(N_dim_permutations.dict_from_fname(file_name))
            txt_path = '/'.join([cwd_path, dire, file_name])
            array = array_from_txt(txt_path)
            frontiers.append(array)
    return params, frontiers

class Wise_permutations(object):
    def __init__(self, poss_values):
        self.poss_values = poss_values
        self.limits = [len(value) for key, value in poss_values.items()]
        self.keys = [key for key, value in poss_values.items()]
        self.vector = np.zeros([len(poss_values)], dtype=int)
        self.num_comb = np.product([len(value) for key, value in poss_values.items()])
        self.first_time = True

    
    def __iter__(self):
        return self

    def __next__(self):
        if self.first_time:
            self.first_time = False
            return [lista[0] for key, lista in self.poss_values.items()]   
        else:
            self.vector[-1] += 1
            i = 1
            while self.vector[-i] >= self.limits[-i]:
                if i + 1 == len(self.vector) and self.vector[0] == self.limits[0]-1:
                    raise StopIteration
                self.vector[-i] = 0
                self.vector[-(i+1)] += 1
                i += 1
            return self.get_params_from_vec()
    
    def string_params(self, lista_params):
        def join_lists(lista, join_char=''):
            return join_char.join(map(str, lista))
        assert(len(self.keys) == len(lista_params))
        param_value = [str(k) +'' + str(p) for k, p in zip(self.keys, lista_params)]
        return join_lists(param_value, '__')

    def get_params_from_vec(self):
        return [values[i2] for i2, values in zip(self.vector, self.poss_values.values())]

class N_dim_permutations(object):
    @classmethod
    def dict_from_fname(cls, fname):
        fname = fname[:-4] #rmv .txt
        strings = fname.split('__')
        dictionary = {}
        for param_value in strings:
            sepi = (re.findall('[0-9\.]',param_value))[0]
            index = param_value.find(sepi)
            param, value = param_value[:index], float(param_value[index:])
            dictionary[param] = value
        return dictionary

def get_ordered_params(parameters):
    columns = np.zeros(len(parameters))
    i = 0
    for __, b in sorted(parameters.items()):
        columns[i] = b
        i+=1
    return columns

def write_non_dominated_frontiers(time, files_names, possible_values, dir_path):
    char_n_file_gen = Wise_permutations(files_names)
    for file_tp, n in char_n_file_gen:
        # Create folder and move there if not exists
        os.chdir(dir_path)
        create_folder(f'{n}{file_tp}')

        print(f'\n\n -------- {n}{file_tp} --------- \n\n')
        os.chdir(f'{dir_path}{n}{file_tp}/')

        fp = f'~/Dropbox/PI/PI2/data/n{n}q10{file_tp}.dat'
        file_stations = util.read_file(fp)
        stations = read_stations(file_stations) # list of stations
        Sol.set_stations(stations)
        
        # For every file let us calculate every frontier
        tuples_combinations = Wise_permutations(possible_values)
        for params in tuples_combinations:
            print('params --> ', params)
            # Train model
            n_pob, ratio_sons, ratio_mutation, num_random_sols = params
            solutions = SolCollection(n_pob=n_pob, ratio_sons=ratio_sons,
                                         ratio_mutation=ratio_mutation, num_random_sols=num_random_sols)
            non_dom_result = solutions.train_time(time)
            # Save file
            out_file_name = tuples_combinations.string_params(params) + '.txt'
            np.savetxt(out_file_name ,non_dom_result)

def write_excel_of_metrics(file_name, directories, DIR_RES, DIR_PATH):
    # Writing EXCEL file with EXCEL sheet with results per folder of data
    with pd.ExcelWriter(DIR_RES + file_name) as writer:
        for dire in directories:
            param_list, frontiers_list = read_frontiers_infolder(DIR_PATH, dire)
            concat_frontiers = np.concatenate(frontiers_list)
            par_front = get_paretto(concat_frontiers)
            limites = np.zeros([2, 2])
            limites[:, 0] ,limites[:, 1] = np.min(par_front, axis=0), np.max(par_front, axis=0)
            par_front = normalized(par_front, limites)

            data = []
            for params, frontier in zip(param_list, frontiers_list):
                n_frontier = normalized(frontier, limites)
                # results every method
                m1 = set_cov(par_front, n_frontier)
                m2 = gen_distance(n_frontier, par_front)
                m3 = spacing(n_frontier)
                m4 = eucl_sum(n_frontier)
                d_row = [*get_ordered_params(params), m1, m2, m3, m4]
                data.append(d_row)

            parameters_names = ['n_pob', 'ratio_sons', 'ratio_mutation', 'num_random_sols', 'set_coverage',\
                        'Gen_dist', 'spacing', 'euclidean']
            df_data = pd.DataFrame(data, columns=parameters_names)
            # fl_name = DIR_RES + '/results/' + dire + '.xlsx'
            df_data.to_excel(writer, sheet_name=dire ,index=False)
        writer.save()

def write_excel_consolidated(fp_consol, directories, DIR_RES):
    # Writing EXCEL file with EXCEL sheet with RESULTS CONSOLIDATED
    with pd.ExcelWriter(DIR_RES + fp_consol, mode='w') as writer:
        all_metrics = []
        m_names = ['set_coverage', 'Gen_dist', 'spacing', 'euclidean']
        n_met = len(m_names)
        for dire in directories:
            table = pd.read_excel(DIR_RES  + 'all_results.xlsx', sheet_name=dire)
            # Promedio las fronteras con diferentes rt_non_dominance ya que
            # por estos momentos local search no se hacia.
            parameters = table.values[:,:-n_met]
            raw_metrics = table.values[:,-n_met:]
            n, p = raw_metrics.shape
            if dire == directories[0]:
                last_parameters = parameters
            assert np.all(parameters == last_parameters)
            last_parameters = parameters
            all_metrics.append(raw_metrics)
        all_metrics = np.array(all_metrics)
        par_names = table.columns[:-n_met]
        par_table = parameters
        medias = np.mean(all_metrics, axis=0)

        # Normalize values
        lim = np.zeros([4,4])
        lim[:, 0] ,lim[:, 1] = np.min(medias, axis=0), np.max(medias, axis=0)
        norm_medias = normalized(medias, lim)

        # Create table
        cols_names = [*['mean ' + n for n in m_names],
                        *['n_means ' + n for n in m_names]]
        table = np.concatenate([medias, norm_medias], axis=1)
        df_table = pd.DataFrame(table, columns=cols_names)

        # Ponderation values
        list_pond = [.7, .2, .05, .05]
        df_table['ponder'] = 0
        names_cols = ['n_means ' + n for n in m_names]
        for a, n_cv_name in zip(list_pond, names_cols):
            df_table['ponder'] += a * df_table[n_cv_name]

        # Writing original table of metric ponderation
        df_table.to_excel(writer, sheet_name='Comparation_all')

        # Write table of the comparations sorted
        df_table.sort_values('ponder', inplace=True, ascending=False)
        df_table.to_excel(writer, sheet_name='sorted_comparation')

        # Write the parameters comparation
        par_table = pd.DataFrame(par_table, columns=par_names)
        par_table.to_excel(writer, sheet_name='Parameters combination')
        
        # To decide which ones
        selected_params = df_table.index[:5]

        print(par_table.loc[selected_params].values)

        aleatory = par_table.sample(5)
