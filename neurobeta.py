from nb_functions import *

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

if __name__ == "__main__":
    i,o = args_checker(*sys.argv)
    print(f"Input file: {i}")
    print(f"Output directory: {o}")
    os.chdir(o)
    print(f'Changed working directory to {o}')
    flirted_file = run_dartel(i, o)
    mwc1t1_checker(flirted_file)
    dicv_outfile = dicv_file_producer(o)
    roi_stdout = roi_xtractor(dicv_outfile)
    print(roi_stdout)
    gm_visualiser(output_dir = o, cleaned_rois = roi_stdout)
    linreg_results, coeff_df = linear_spatial_regression(input_file = i, cleaned_rois = roi_stdout, output_dir = o)
    ml_results = machine_learner(coeff_df, input_file = i, output_dir = o)
    report_gen(infile = i, output_dir = o, linreg_results = linreg_results, ml_results = ml_results)




