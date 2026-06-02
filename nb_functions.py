import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import nilearn
import click
import os
import sys
import nipype
import subprocess

def nb_exit(code):
    print(f"Exiting Neurobeta. Code: {code}")
    sys.exit(code)

def nb_greet():
    print("Neurobeta v0.1 - performing linear spatial regressions with GM abonormality data and neurotransmitter density")
    print("Usage: neurobeta.py <input_file> <output_dir>")
    print("Use 'neurobeta.py -h' for more information.")
    nb_exit(0)

def nb_help():
    print("This program aims to take a T1-weighted MRI image, and then:" \
    "\n1. Obtain a preprocessed GM map z-scored to a healthy control group" \
    "\n2. Perform a linear regression between the GM map and neurotransmitter density maps, to obtain a set of regression coefficients" \
    "\n3. Obtain statistics for the coefficients (analysis of regression, machine learning performance) which are then" \
    "\n4. Saved in a .txt file (beta coefficients) and a html report file in the output directory")
    print("\nUsage: neurobeta.py <input_file> <output_dir>")
    print("Note: Input file MUST EXISTS. Output directory will be created if it does not exist.")



def args_checker(*args):
    if len(args) == 1: # No arguments provided
        nb_greet()
    elif len(args) > 1: # Only one argument 
        if '-h' in args:
            nb_help()
        if len(args) == 3:
            input_file = args[1]
            output_dir = args[2]
            # Check if input file exists and is a valid NIfTI file
            if not os.path.exists(input_file) or not input_file.endswith('.nii.gz'):
                print(f"Input file {input_file} does not exist or is not a valid NIfTI file. Please provide a valid input file.")
                nb_exit(1)
            # Check if output directory exists, if not create it
            if not os.path.isdir(output_dir):
                print(f"Output directory {output_dir} does not exist. Creating it now.")
                os.makedirs(output_dir, exist_ok=True)
                if not os.path.isdir(output_dir):
                    print(f"Failed to create output directory {output_dir}. Please check permissions and try again.")
                    nb_exit(1)
            return input_file, output_dir
        else:
            print("Invalid number of arguments. Please provide exactly 2 arguments: <input_file> <output_dir>")
            nb_exit(1)

def nb_getbasename(infile):
    nb_basename = os.path.basename(infile).replace('.nii.gz', '')
    print(f'Basename for any intermediate files: {nb_basename}')
    return nb_basename


def brain_extractor(input_file, output_dir, nb_basename):
    os.chdir(output_dir) # Change working directory to output directory to save outputs there
    print(f'Changed working directory to {output_dir}')
    print(f"Step 1: Extracting brain from input file {input_file}...")
    # Here you would add the code to perform brain extraction using FSL's BET or another tool
    # For example, using nipype to interface with FSL:
    from nipype.interfaces import fsl
    bet_file = f'{output_dir}/step1_{nb_basename}_brain_extracted.nii.gz'
    bet = fsl.BET(in_file=input_file, out_file=bet_file, mask=False, frac = 0.35)
    bet.run()
    if not os.path.exists(bet_file):
        print(f"Brain extraction failed. Output file not found: {bet_file}")
        nb_exit(1)
    else:
        print(f"Brain extraction successful. Output file: {bet_file}")
        return bet_file

def segmenter(bet_file, nb_basename):
    if os.path.exists(bet_file) == False:
        print(f"Segmentation failed. Brain extracted file not found: {bet_file}")
        nb_exit(1)
    else:
        print(f"Step 2: Segmenting brain extracted file {bet_file} into GM, WM and CSF...")
        import subprocess
        output_basename = f"step2_{nb_basename}_segmented"        
        cmd_fast = f"fast -n 3 -v -o {output_basename} {bet_file}"
        subprocess.run(cmd_fast, shell=True)
        if not os.path.exists(f'{output_basename}_pve_1.nii.gz'):
            print(f"Segmentation failed. Output file not found: {output_basename}_pve_1.nii.gz")
            nb_exit(1)
        else:
            print(f"Segmentation successful. Output files: {output_basename}_pve_0.nii.gz (CSF), {output_basename}_pve_1.nii.gz (GM), {output_basename}_pve_2.nii.gz (WM)")
            return f'{output_basename}_pve_1.nii.gz' # Return GM partial volume estimate file


def coregister(infile, bet_file, nb_basename, dof = 6):
    # bet_file is to get an ICV
    if os.path.exists(infile) == False:
        print(f"Coregistration failed. Input file not found: {infile}")
        nb_exit(1)
    else:
        print(f"Step 3: Coregistering file {infile} to MNI 1mm3 space...")
        """
        FLIRT to MNI152
        """
        mni152_brain_reference= '/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/MNI152_T1_1mm_brain.nii.gz'
        flirted_file=f"step3_{nb_basename}_coregistered.nii.gz" 
        cmd_flirt = f"flirt -in {infile} -ref {mni152_brain_reference}  -dof {dof} -applyisoxfm 1.0 -interp nearestneighbour -out {flirted_file}"
        
        subprocess.run(cmd_flirt, shell=True)
        if not os.path.exists(flirted_file):
            print(f"Coregistration failed. Output file not found: {flirted_file}")
            nb_exit(1)
        else:
            print(f"Coregistration successful. Output file: {flirted_file}")
            print("Calculating ICV...")
            import nibabel as nib
            import numpy as np
            img = nib.load(bet_file)
            data = img.get_fdata()
            voxel_volume = np.prod(img.header.get_zooms()[:3])
            brain_mask = data > 0
            icv = np.sum(brain_mask) * voxel_volume
            print(f"Estimated intracranial volume (ICV): {icv:.2f} mm3")
            """
            normalise GM scan
            """
            mni152_brainmask_reference= "/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/MNI152_T1_1mm_brain_mask.nii.gz"
            norm_filename = f"step4_{nb_basename}_norm.nii.gz"
            print("Step 4: Normalising with scan mean and standard deviation...")

 
            cmd_mean = f"fslstats {flirted_file} -k {mni152_brainmask_reference} -m"
            res_mean = subprocess.run(cmd_mean, text = True, shell = True, capture_output = True)
            cmd_std = f"fslstats {flirted_file} -k {mni152_brainmask_reference} -s"
            res_std = subprocess.run(cmd_std, text = True, shell = True, capture_output = True)
            print(f"Mean for z-scoring: {res_mean.stdout}. Standard deviation for z-scoring: {res_std.stdout}")
            norm_cmd = f"fslmaths {flirted_file} -sub {res_mean.stdout.strip()} -div {res_std.stdout.strip()} -mas {mni152_brainmask_reference} {norm_filename}"
            subprocess.run(norm_cmd, shell = True)
            if os.path.exists(norm_filename) == False:
                print(f"Coregistration failed. Output file not found: {norm_filename}")
                nb_exit(1)
            else:
                print("Step 5: Normalising with ICV this time...")
                dicv_outfile = f"step5_{nb_basename}_dicv.nii.gz"
                dicv_cmd = f"fslmaths {norm_filename} -div {icv} {dicv_outfile}"
                subprocess.run(dicv_cmd, shell = True)
                if os.path.exists(dicv_outfile) == False:
                    print(f"Coregistration failed. Output file not found: {dicv_outfile}")
                    nb_exit(1)
                else:
                    print(f"Success: {dicv_outfile} located and ready for ROI analysis")
                    return dicv_outfile

def roi_xtractor(dicv_outfile, nb_basename):
    pass

