import synthetic_exp as se
import sys
from snmfem.conf import DATASETS_PATH, RESULTS_PATH
from pathlib import Path
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np
import json

data_folder = sys.argv[1]
json_file = sys.argv[2]
folder_path = DATASETS_PATH / Path(data_folder)
data_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
full_save_dict = {}
list_angles = []
list_mse = []

for file in tqdm(data_files) :
    file_path = Path(data_folder) / Path(file)
    angles, mse = se.run_experiment(json_file,save=False,data_filename = file_path) 
    list_angles.append(angles)
    list_mse.append(mse)
    full_save_dict[str(file_path)] = { "angles" : angles, "mse" : mse}

array_angles = np.array(list_angles)
array_nse = np.array(list_mse)
stat_dict = {}
for i in range(array_angles.shape[1]) : 
    str_avg = "avg phase {}".format(i)
    str_std = "std phase {}".format(i)
    stat_dict[str_avg] = np.average(array_angles,axis=0)[i]
    stat_dict[str_std] = np.std(array_angles,axis=0)[i]

full_save_dict["stats"] = stat_dict
save_path = RESULTS_PATH / Path("summaries/" + json_file[:-4] + "_batch.json")

with open(save_path, 'w') as outfile:
    json.dump(full_save_dict, outfile, sort_keys=True, indent=4)