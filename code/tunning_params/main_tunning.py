import os
from tunning_library import write_excel_of_metrics, \
            write_excel_consolidated, write_non_dominated_frontiers, ls

DIR_PATH = '~/Dropbox/PI/ShortGenetic/code/tunning_params/frontiers/'
DIR_RES = '~/Dropbox/PI/ShortGenetic/code/tunning_params/'
time = 10 # 3600/ 24/ 56#
files_names = {
    'files_type': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], # 'A'
    'files_n': ['20', '30', '40', '50', '60', '100', '300']
}

possible_values = {
    'n_pob': [20, 40, 80],
    'ratios_sons': [.5 ,.75, 1],
    'ratio_mutation': [.05, .2, .6],
    'num_random_sols': [4, 10, 15]
}

# Create folders with the non dominated frontiers
write_non_dominated_frontiers(time, files_names, possible_values, DIR_PATH)

# Write excel of metrics
os.chdir(DIR_PATH)
directories = ls()
write_excel_of_metrics(file_name='all_results.xlsx', directories=directories, DIR_RES=DIR_RES, DIR_PATH=DIR_PATH)   

# Write means consolidated with ponderated metrics and write them in means.xlsx
write_excel_consolidated('means.xlsx', directories, DIR_RES)
