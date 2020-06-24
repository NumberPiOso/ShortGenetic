
import os

TIME_LIMIT = None

def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# 1. ask folder of mps
mps_dir = input('Which is the name of the folders in which .mps are contained?\n')

if not os.path.isdir(mps_dir):
    print("\n-- Folder specified does not exists --\n\n")
    raise ValueError

name_model_dir = mps_dir.split('_')[1]
mps_models = os.listdir(mps_dir)

# 2. create if necessary folder of log and results
res_dir = 'res_' + name_model_dir
log_dir = 'log_' + name_model_dir
create_folder(res_dir)
create_folder(log_dir)

# 3. Create command as string
file_body = ""
for file_mps in sorted(mps_models):
    mps_model = file_mps.split('.')[0]
    if TIME_LIMIT:
        comm = f"gurobi_cl ResultFile={res_dir}/{mps_model}.sol  TimeLimit={TIME_LIMIT} {mps_dir}/{mps_model}.mps LogFile={log_dir}/{mps_model}.log\n"
    else:
        comm = f"gurobi_cl ResultFile={res_dir}/{mps_model}.sol  {mps_dir}/{mps_model}.mps LogFile={log_dir}/{mps_model}.log\n"
    file_body = file_body + comm

# 3. Write .sh
mod_files = open(f"run_model_{name_model_dir}.sh", "w")
n = mod_files.write(file_body)
mod_files.close()