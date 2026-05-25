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
    

def brain_extractor(input_file, output_dir):
    os.chdir(output_dir) # Change working directory to output directory to save outputs there
    print(f"Extracting brain from input file {input_file}...")
    # Here you would add the code to perform brain extraction using FSL's BET or another tool
    # For example, using nipype to interface with FSL:
    from nipype.interfaces import fsl
    bet_file = f'{output_dir}/{os.path.basename(input_file).replace(".nii.gz", "")}_brain_extracted.nii.gz'
    bet = fsl.BET(in_file=input_file, out_file=bet_file, mask=False, frac = 0.35)
    bet.run()
    if not os.path.exists(bet_file):
        print(f"Brain extraction failed. Output file not found: {bet_file}")
        nb_exit(1)
    else:
        print(f"Brain extraction successful. Output file: {bet_file}")
        return bet_file

def segmenter(bet_file):
    if os.path.exists(bet_file) == False:
        print(f"Segmentation failed. Brain extracted file not found: {bet_file}")
        nb_exit(1)
    else:
        print(f"Segmenting brain extracted file {bet_file} into GM, WM and CSF...")
        from nipype.interfaces import fsl
        fast = fsl.FAST(in_files=bet_file, out_basename=f'{os.path.basename(bet_file).replace("_brain_extracted.nii.gz", "")}_segmented', verbose = True, output_type='NIFTI_GZ')
        fast.run()
        if not os.path.exists(f'{os.path.basename(bet_file).replace("_brain_extracted.nii.gz", "")}_segmented_pve_1.nii.gz'):
            print(f"Segmentation failed. Output file not found: {os.path.basename(bet_file).replace('_brain_extracted.nii.gz', '')}_segmented_pve_1.nii.gz")
            nb_exit(1)
        else:
            print(f"Segmentation successful. Output files: {os.path.basename(bet_file).replace('_brain_extracted.nii.gz', '')}_segmented_pve_0.nii.gz (CSF), {os.path.basename(bet_file).replace('_brain_extracted.nii.gz', '')}_segmented_pve_1.nii.gz (GM), {os.path.basename(bet_file).replace('_brain_extracted.nii.gz', '')}_segmented_pve_2.nii.gz (WM)")
            return f'{os.path.basename(bet_file).replace("_brain_extracted.nii.gz", "")}_segmented_pve_1.nii.gz' # Return GM partial volume estimate file


def coregister(infile):
    if os.path.exists(infile) == False:
        print(f"Coregistration failed. Input file not found: {infile}")
        nb_exit(1)
    else:
        print(f"Coregistering file {infile} to MNI 1mm3 space...")
        """
        FLIRT to MNI152
        """
        
        from nipype.interfaces import fsl
        flirt = fsl.FLIRT(in_file=infile, 
                          reference= '/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/MNI152_T1_1mm_brain.nii.gz', 
                          out_file=f'{os.path.basename(infile).replace(".nii.gz", "")}_coregistered.nii.gz', 
                          apply_isoxfm = 1.0, 
                          verbose = True, 
                          output_type='NIFTI_GZ'
                          )
        flirt.run()
        if not os.path.exists(f'{os.path.basename(infile).replace(".nii.gz", "")}_coregistered.nii.gz'):
            print(f"Coregistration failed. Output file not found: {os.path.basename(infile).replace('.nii.gz', '')}_coregistered.nii.gz")
            nb_exit(1)
        else:
            print(f"Coregistration successful. Output file: {os.path.basename(infile).replace('.nii.gz', '')}_coregistered.nii.gz")
            flirted_file = f'{os.path.basename(infile).replace(".nii.gz", "")}_coregistered.nii.gz'
            import nibabel as nib
            import numpy as np
            img = nib.load(flirted_file)
            data = img.get_fdata()
            voxel_volume = np.prod(img.header.get_zooms()[:3])
            brain_mask = data > 0
            icv = np.sum(brain_mask) * voxel_volume
            print(f"Estimated intracranial volume (ICV): {icv:.2f} mm3")
            """
            normalise GM scan
            """
            import scipy
            from scipy.stats import zscore
            brain_values = data[brain_mask]
            brain_values_z = zscore(brain_values) # Divide z-scored data by ICV to normalise for head size
            norm_data = np.zeros_like(data)
            norm_data[brain_mask] = brain_values_z / icv
            dicv_outfile = f'{os.path.basename(infile).replace(".nii.gz", "")}_dicv.nii.gz'

            dicv_img = nib.Nifti1Image(
                norm_data.astype(np.float32),
                img.affine,
                img.header
            )

            nib.save(dicv_img, dicv_outfile)
            flirt = fsl.FLIRT(in_file=dicv_outfile, 
                    reference= '/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/MNI152_T1_1mm_brain.nii.gz', 
                    out_file=f'{os.path.basename(infile).replace(".nii.gz", "")}_reflirted.nii.gz', 
                    verbose = True, 
                    output_type='NIFTI_GZ'
                    )
            flirt.run()

            print(f"Saved normalized image: {os.path.basename(infile).replace('.nii.gz', '')}_reflirted.nii.gz")

            return dicv_outfile
            


