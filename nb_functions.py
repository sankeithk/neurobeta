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
import deepmriprep 
from deepmriprep import run_preprocess

# General purpose functions

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

def nb_getbasename(infile, verbose = False):
    nb_basename = os.path.basename(infile).replace('.nii.gz', '')
    if verbose:
        print(f'Basename for any intermediate files: {nb_basename}')
    return nb_basename

def args_checker(*args):
    if len(args) == 1: # No arguments provided
        nb_greet()
    elif len(args) > 1: # Only one argument 
        if '-h' in args:
            nb_help()
        if len(args) == 3:
            input_file = args[1]
            output_dir = args[2]
            nb_getbasename(input_file, verbose = True)
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



## DARTEL function(s) - WIP

def run_dartel(input_file, output_dir):
    import os
    import subprocess
    # Configuration: Note from Sankeith - please change these paths for your system
    spm_launcher = '/home/sankeith/spm_standalone/run_spm25.sh'
    mcr_path = '/home/sankeith/MATLAB_Runtime/R2024b'
    spm_root = '/home/sankeith/spm_standalone'
    # For the TPM file path, you need to change it within matlab_template. I've made a 'dud' tpm_file variable here for readability
    tpm_file = '/home/sankeith/spm_standalone/spm25_mcr/spm25/tpm/TPM.nii'

    # Use fslreorient2std to make sure input image is in correct orientation
    import subprocess
    reorient2std_outpath = os.path.join(output_dir,f'{nb_getbasename(input_file)}_reoriented_fsl.nii.gz')
    cmd_reorient = f"fslreorient2std {input_file} {reorient2std_outpath}"
    print('Reorienting input image ahead of segmentation...')
    subprocess.run(cmd_reorient, shell=True)


    # Prepare paths for MATLAB
    m_input = reorient2std_outpath.replace('\\', '/')
    m_output = output_dir.replace('\\', '/')
    m_spm_root = spm_root.replace('\\', '/')

    matlab_template = """
try
    % 1. INITIALIZE
    t1_file = 'VAR_INPUT';
    out_dir = 'VAR_OUTPUT';
    spm_dir = 'VAR_SPMROOT';
    
    cd(out_dir);
    spm('defaults', 'FMRI');
    spm_jobman('initcfg');

    % 2. HANDLE COMPRESSED INPUT (.nii.gz)
    [~, t1_base, t1_ext] = fileparts(t1_file);
    if strcmpi(t1_ext, '.gz')
        fprintf('Decompressing .nii.gz file...\\n');
        decompressed_files = gunzip(t1_file, out_dir);
        t1_out = decompressed_files{1};
    else
        t1_out = fullfile(out_dir, [t1_base t1_ext]);
        if ~strcmp(t1_file, t1_out)
            copyfile(t1_file, t1_out);
        end
    end

    % 3. PATH CHECK FOR TPM
    tpm_file = '/home/sankeith/spm_standalone/spm25_mcr/spm25/tpm/TPM.nii';
    if ~exist(tpm_file, 'file')
        tpm_file = '/home/sankeith/spm_standalone/spm25_mcr/spm_25/tpm/TPM.nii';
    end
    if ~exist(tpm_file, 'file')
        error('TPM.nii not found. Please check the path inside neurobeta.py');
    end

    % 4. STEP 1: SEGMENTATION
    % Note: native = [1 1] creates both c* (native) and rc* (DARTEL-imported)
    fprintf('Step 1: Segmentation...\\n');
    ngaus_vals = [1 1 2 3 4 2];
    for k = 1:6
        tissue(k).tpm    = {sprintf('%s,%d', tpm_file, k)};
        tissue(k).ngaus  = ngaus_vals(k); 
        tissue(k).native = [1 1]; 
        tissue(k).warped = [0, 0];
    end

    matlabbatch{1}.spm.spatial.preproc.channel.vols     = {[t1_out ',1']};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg  = 0.001;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{1}.spm.spatial.preproc.channel.write    = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue           = tissue;
    matlabbatch{1}.spm.spatial.preproc.warp.mrf         = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup     = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.reg         = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg      = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm        = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.samp        = 3;
    matlabbatch{1}.spm.spatial.preproc.warp.write       = [1 1];
    
    spm_jobman('run', matlabbatch);
    clear matlabbatch;

    % 5. STEP 2: DARTEL WARPING (Create Template)
    % We skip the "Import" step because rc* files were created in Step 1.
    fprintf('Step 2: DARTEL Warping...\\n');
    rc1_files = cellstr(spm_select('FPList', out_dir, '^rc1.*\\.nii$'));
    rc2_files = cellstr(spm_select('FPList', out_dir, '^rc2.*\\.nii$'));
    
    if isempty(rc1_files{1}), error('DARTEL-imported files (rc1) not found!'); end

    matlabbatch{1}.spm.tools.dartel.warp.images = {rc1_files; rc2_files};
    matlabbatch{1}.spm.tools.dartel.warp.settings.template = 'Template';
    matlabbatch{1}.spm.tools.dartel.warp.settings.rform    = 0;
    matlabbatch{1}.spm.tools.dartel.warp.settings.optim.lmreg = 0.01;
    matlabbatch{1}.spm.tools.dartel.warp.settings.optim.cyc   = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.optim.its   = 3;
    spm_jobman('run', matlabbatch);
    clear matlabbatch;

    % 6. STEP 3: NORMALISE TO MNI
    fprintf('Step 3: Normalising to MNI...\\n');
    u_rc1 = cellstr(spm_select('FPList', out_dir, '^u_rc1.*\\.nii$'));
    tmpl  = cellstr(spm_select('FPList', out_dir, '^Template_6\\.nii$'));
    c1    = cellstr(spm_select('FPList', out_dir, '^c1.*\\.nii$'));
    
    matlabbatch{1}.spm.tools.dartel.mni_norm.template = tmpl;
    matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj.flowfield = u_rc1;
    matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj.images    = c1;
    matlabbatch{1}.spm.tools.dartel.mni_norm.vox      = [1 1 1];
    matlabbatch{1}.spm.tools.dartel.mni_norm.bb       = [-90 -126 -72; 90 90 108];
    matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 1; % Modulated
    matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm     = [0 0 0];
    spm_jobman('run', matlabbatch);
    clear matlabbatch;

    % 7. STEP 4: Z-SCORE & COMPRESSION
    fprintf('Step 4: Z-scoring results...\\n');
    mwc1_list = spm_select('FPList', out_dir, '^mwc1.*\\.nii$');
    mwc1_path = deblank(mwc1_list(1,:));
    mwc1_std  = fullfile(out_dir, 'mwc1t1.nii');
    movefile(mwc1_path, mwc1_std);

    V = spm_vol(mwc1_std);
    img = spm_read_vols(V);
    mask = img > 0 & isfinite(img);
    mu = mean(img(mask));
    sd = std(img(mask));
    img_z = (img - mu) / sd;
    img_z(~mask) = 0;
    
    Vz = V; 
    Vz.fname = fullfile(out_dir, 'mwc1t1_zscore.nii'); 
    Vz.dt = [16 0];
    spm_write_vol(Vz, img_z);
    
    fprintf('Compressing files...\\n');
    gzip(mwc1_std); 
    gzip(Vz.fname);
    delete(mwc1_std); 
    delete(Vz.fname);

    fprintf('--- PIPELINE FINISHED SUCCESSFULLY ---\\n');

catch ME 
    fprintf('!! ERROR: %s\\n', ME.message); 
    if ~isempty(ME.stack)
        fprintf('!! Error in %s at line %d\\n', ME.stack(1).name, ME.stack(1).line); 
    end
    exit(1); 
end 
exit(0); 
"""

    full_code = matlab_template.replace('VAR_INPUT', m_input)\
                               .replace('VAR_OUTPUT', m_output)\
                               .replace('VAR_SPMROOT', m_spm_root)

    runner_path = os.path.join(output_dir, 'pipeline_runner.m')
    with open(runner_path, 'w') as f:
        f.write(full_code)

    cmd = [spm_launcher, mcr_path, 'script', runner_path]
    print(f"--- Launching SPM25 Standalone Pipeline ---")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()

def mwc1t1_checker(output_dir):
    if os.path.exists(os.path.join(output_dir, 'mwc1t1_zscore.nii.gz')):
        mwc1t1_file = os.path.join(output_dir, 'mwc1t1_zscore.nii.gz')
        print(f'{mwc1t1_file} exists')
        print(f'Checking quality of output - it\'s also good to look at the file yourself')
        import nibabel as nib
        import numpy as np
        img = nib.load(mwc1t1_file)
        data = img.get_fdata()
        voxel_volume = np.prod(img.header.get_zooms()[:3])
        print(f'Voxel volume: {voxel_volume}')
        assert voxel_volume == 1.0
        cmd_voxmean = f'fslstats {mwc1t1_file} -M'
        cmd_voxstdev = f'fslstats {mwc1t1_file} -S'
        voxmean = subprocess.run(cmd_voxmean, capture_output=True, text=True, shell = True)
        voxstdev = subprocess.run(cmd_voxstdev, capture_output=True, text=True, shell  = True)
        print(f'Mean of non-zero voxels for {mwc1t1_file}: {voxmean.stdout.strip()}')
        print(f'Standard deviation of non-zero voxels for {mwc1t1_file}: {voxstdev.stdout.strip()}')
    else:
        print('Error - mwc1t1 file not found')
        nb_exit(2)

def dicv_file_producer(output_dir):
    pass


# import gzip
# import shutil

# def run_dartel(input_file, output_dir):
#     # 1. SPM needs .nii, not .nii.gz
#     if input_file.endswith('.gz'):
#         unzipped_file = os.path.join(output_dir, os.path.basename(input_file).replace('.gz', ''))
#         print(f"Unzipping {input_file} to {unzipped_file}...")
#         with gzip.open(input_file, 'rb') as f_in:
#             with open(unzipped_file, 'wb') as f_out:
#                 shutil.copyfileobj(f_in, f_out)
#         input_file = unzipped_file

#     input_file = str(os.path.abspath(input_file))
#     output_dir = str(os.path.abspath(output_dir))
    
#     # 2. Path to your .m scripts
#     script_dir = "/mnt/c/Users/User/Downloads/neurobeta"
    
#     command = ["/mnt/c/Users/User/Downloads/neurobeta/run_dartel.sh", input_file, output_dir, script_dir]
    
#     print('Starting DARTEL-based GM segmentation...')
#     try:
#         # Note: Changed to capture output more effectively for debugging
#         result = subprocess.run(command, check=True, text=True, capture_output=True)
#         print(result.stdout)
#         print(result.stderr)

#         print("--- Pipeline Finished Successfully ---")
#     except subprocess.CalledProcessError as e:
#         print("--- Pipeline FAILED ---")
#         print("STDOUT:", e.stdout) # Crucial: SPM errors usually print to stdout
#         print("STDERR:", e.stderr)
#         nb_exit(2)



## FSL preprocessing steps

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

