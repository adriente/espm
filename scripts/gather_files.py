from glob import glob
import os
import snmfem.experiments as e
import sys

target = sys.argv[1]

os.chdir(target)
files = glob("*{}*.txt".format(target))
e.gather_results(files,target+"_gathered.txt")