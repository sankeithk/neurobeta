import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import nilearn
import click
import os
import sys
import subprocess
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
    if os.path.exists(os.path.join(output_dir, 'mwc1t1_zscore.nii.gz')):
        mwc1t1_file = os.path.join(output_dir, 'mwc1t1_zscore.nii.gz')
        mni152_brain_reference= '/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/MNI152_T1_1mm_brain.nii.gz'
        flirted_file=os.path.join(output_dir, "mwc1t1_zscore_flirted.nii.gz")
        cmd_flirt = f"flirt -v -in {mwc1t1_file} -ref {mni152_brain_reference}  -dof 12 -applyisoxfm 1.0 -interp nearestneighbour -out {flirted_file}"
        subprocess.run(cmd_flirt, shell=True)
        if os.path.exists(flirted_file):
            print('Fully processed probabilistic map (not normalised by ICV yet) successfully created. Moving onto mwc1t1 checker')
            return flirted_file
        else:
            print('Error - Fully processed probabilistic map (not normalised by ICV yet) not found')
            nb_exit(2)
    

def mwc1t1_checker(flirted_file):
    if os.path.exists(flirted_file):
        mwc1t1_file = flirted_file
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
        print('Error - Fully processed probabilistic map (not normalised by ICV yet) not found')
        nb_exit(2)

def dicv_file_producer(output_dir):
    os.chdir(output_dir)
    if os.path.exists(os.path.join(output_dir, "mwc1t1_zscore_flirted.nii.gz")) == False:
        nb_exit(3)
    else:
        print(f'ICV normalisation starting. Changed working directory to {output_dir}')
        if os.path.exists(os.path.join('/mnt/c/Users/User/Downloads/neurobeta', 'dicv_file_producer.sh')):
            cmd_dicv = f"chmod +x /mnt/c/Users/User/Downloads/neurobeta/dicv_file_producer.sh"
            subprocess.run(cmd_dicv, shell = True)
            cmd_dicv = f"bash /mnt/c/Users/User/Downloads/neurobeta/dicv_file_producer.sh {output_dir}"
            subprocess.run(cmd_dicv, shell = True)
            if os.path.exists('icv.txt'):
                with open('icv.txt') as f:
                    icv = f.readline()
                    icv = int(icv)
                    dicv_outfile = f"mwc1t1_zscore_flirted_dicv.nii.gz"
                    dicv_cmd = f"fslmaths mwc1t1_zscore_flirted.nii.gz -div {icv} {dicv_outfile}"
                    subprocess.run(dicv_cmd, shell = True)
                    if os.path.exists(dicv_outfile):
                        print('ICV normalisation successful!')
                        return dicv_outfile
                    else:
                        print('Error - File not found')
                        nb_exit(2)

def roi_xtractor(dicv_outfile,
                 hc_mean = '/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/scott_10k_cn_tmean.nii.gz',
                 hc_std = '/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/scott_10k_cn_tstd.nii.gz',
                 mni152_brainmask_reference= "/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/MNI152_T1_1mm_brain_mask.nii.gz",
                 atlas_path = "/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/gm_only_MNI152_1mm_desikan+aseg.nii.gz"):
    if os.path.exists(dicv_outfile):
        zscore_dicv_outfile = 'mwc1t1_zscore_flirted_dicv_atrophymap.nii.gz'
        print('Z-scoring against healthy controls...')
        zscore_cmd = f"fslmaths {dicv_outfile} -sub {hc_mean} -div {hc_std} -mas {mni152_brainmask_reference} {zscore_dicv_outfile}"
        subprocess.run(zscore_cmd, shell=True)
        if os.path.exists(zscore_dicv_outfile):
            print(f'Successful z-scoring againse standard GM maps. Moving to parcellation with atlas at {atlas_path}...')
            roi_output = 'rois.txt'
            parcellate_cmd = f"fslmeants -i {zscore_dicv_outfile} --label={atlas_path} -o {roi_output}"
            subprocess.run(parcellate_cmd, shell = True)
            import pandas as pd
            df = pd.read_csv('rois.txt', sep='\s+', header=None)
            all_rois = df.values.tolist()[0]
            print(all_rois)
            cleaned_rois = []
            for i in all_rois:
                if i !=float(0):
                    cleaned_rois.append(i)
            assert len(cleaned_rois) == 86
            import pickle
            pickle.dump(cleaned_rois, open('cleaned_rois.pkl', 'wb'))
            return cleaned_rois

        
def gm_visualiser(output_dir, 
                  cleaned_rois, 
                  atlas = "/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/gm_only_MNI152_1mm_desikan+aseg.nii.gz"):
    os.chdir(output_dir)
    os.makedirs('plots', exist_ok = True)
    print(f'Visualising the parcellated GM atrophy - all plots will be stored in {output_dir}/plots...')
    import numpy as np
    cleaned_rois_array = np.array(cleaned_rois)
    import nilearn
    from nilearn import image
    gm_atlas_img = nilearn.image.load_img(atlas)
    from nilearn import maskers, datasets, plotting
    labels_masker = maskers.NiftiLabelsMasker(labels_img=gm_atlas_img, standardize=None)
    labels_masker.fit()
    stat_img = labels_masker.inverse_transform(cleaned_rois_array.reshape(1,-1))
    img_data = stat_img.get_fdata()
    non_zero_mask = img_data != 0
    img_data[non_zero_mask] = img_data[non_zero_mask].reshape(-1, 1).flatten()
    scaled_stat_img = nilearn.image.new_img_like(stat_img, img_data)
    fsaverage = datasets.fetch_surf_fsaverage()
    cmap = plt.get_cmap('coolwarm', 8)
    print(f'Plotting surface view...')
    plotting.plot_img_on_surf(
        stat_map = scaled_stat_img,
        mask_img=None,
        views=['lateral', 'medial'],
        hemispheres=['left'],
        title=f'GM ROI: Surface view',
        colorbar=True,
        vmin=-1,
        vmax=1,
        cmap='coolwarm'
    )
    plt.savefig(f"{output_dir}/plots/surface_view_plots.png", dpi=600)
    print(f'Plotting cross-sectional stat map...')
    plotting.plot_stat_map(
    stat_img,
    display_mode='z',
    vmin=-1,
    vmax=1,
    cut_coords=[0],
    cmap=cmap,
    bg_img=None)
    
    plt.savefig(f"{output_dir}/plots/cross_sectional_statmap.png", dpi=600)

import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import os, pickle,tqdm, neuromaps, netneurotools, nilearn, scipy
from nilearn import datasets, plotting, maskers
from scipy import ndimage, spatial
import nibabel as nib
from tqdm import tqdm
from neuromaps.nulls import moran
from netneurotools import utils
import sys

def get_r_sq(X,y,model):
    model.fit(X, y)
    yhat = model.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (SS_Residual / SS_Total)
    return r_squared
    
def get_adj_r_sq(X, y, model):
    """
    NOTE FROM SANKEITH: THIS IS NOT MINE.
    I TOOK THIS CODE FROM JUSTINE HANSEN THANK YOU JUSTINE HANSEN
    """
    r_squared = get_r_sq(X, y, model)
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return adjusted_r_squared

def get_centroids(img, labels=None, image_space=False):
    """
    NOTE FROM SANKEITH: THIS IS NOT MINE I ADAPTED THIS CODE FROM AN OLD VERSION OF NETNEUROTOOLS
    
    Find centroids of `labels` in `img`.

    Parameters
    ----------
    img : niimg-like object
        3D image containing integer label at each point
    labels : array_like, optional
        List of labels for which to find centroids. If not specified all
        labels present in `img` will be used. Zero will be ignored as it is
        considered "background." Default: None
    image_space : bool, optional
        Whether to return xyz (image space) coordinates for centroids based
        on transformation in `img.affine`. Default: False

    Returns
    -------
    centroids : (N, 3) np.ndarray
        Coordinates of centroids for ROIs in input data
    """ 
    from nilearn.image import load_img, check_niimg_3d

    img = check_niimg_3d(img)
    data = nilearn.image.get_data(img)
    if labels is None:
        labels = np.trim_zeros(np.unique(data))
    centroids = np.vstack(ndimage.center_of_mass(data, labels=data,
                                                    index=labels))
    if image_space:
        centroids = nib.affines.apply_affine(img.affine, centroids)
    return centroids

def get_distmat(atlas):
    """
    Generate distance matrix for the ROIs. 
    The distance matrix essentially indicates relative distances of ROIs to each other.
    The distance matrix is n x n big (n = number of ROIs in an atlas)
    """
    atlas_path = atlas
    gm_coords = get_centroids(atlas_path)
    distmat = scipy.spatial.distance_matrix(gm_coords, gm_coords)
    return distmat

def get_regr_p_val_moran(X, y, atlas, distmat, output_dir, parcellation = None, n_perm = 10000):
    
    ## Get the original r squared
    emp_r_squared = get_r_sq(X, y, LinearRegression(fit_intercept = False))

    ## Selects atlas
    atlas_path = atlas
    if os.path.exists(atlas_path):
        
        #Generate distance matrix for the ROIs. 
        #The distance matrix essentially indicates relative distances of ROIs to each other.
        #The distance matrix is n x n big (n = number of ROIs in an atlas)
        # gm_coords = get_centroids(atlas_path)
        # distmat = scipy.spatial.distance_matrix(gm_coords, gm_coords)
        
        #Nulls of y are generated here
        gm_atlas_img = nilearn.image.load_img(atlas_path)
        labels_masker = maskers.NiftiLabelsMasker(labels_img=gm_atlas_img, standardize=None)
        labels_masker.fit()
        stat_img = labels_masker.inverse_transform(y.reshape(1,-1))
        stat_img.to_filename(f'{output_dir}/for_nulls.nii.gz')
        
        null_y_maps = neuromaps.nulls.moran(data= y.tolist(),
                                            distmat = distmat, # The docs say 'Providing this will cause atlas, density, and parcellation to be ignored.', but for some reason this scipt works with all of them combined idk why exactly
                                            n_perm = n_perm,
                                            atlas='mni152', 
                                            density='1mm', 
                                            parcellation = atlas_path,
                                            seed = 42)
        
        null_y_maps_transposed = np.transpose(null_y_maps)
        null_vals = [get_r_sq(X, null_y.reshape(-1, 1), LinearRegression(fit_intercept=False)) for null_y in null_y_maps_transposed]
    
        pval = (1 + np.sum(null_vals > emp_r_squared)) / (len(null_vals) + 1)
        return pval, emp_r_squared

    else:
        raise FileNotFoundError

def linear_spatial_regression(input_file, cleaned_rois, output_dir):
    """
    This function aims to:

    * Conduct a linear regression between a scan's ROI values and corresponding ones from selected neurotransmitter maps
    * Find the scan's r squared, adjusted r squared and p value
    * Spit out the coefficients into one dataframe and the other stats into another
    """
    #Initial config

    nb_basename = nb_getbasename(input_file)
    neuromaps_df = pd.read_csv('/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/gm_neuromaps_rois.csv', low_memory = False)
    neuromaps_names = neuromaps_df.columns.tolist()
    neuromaps_df_means = neuromaps_df.mean()
    demeaned_neuromaps_df = neuromaps_df - neuromaps_df_means
    X = demeaned_neuromaps_df.to_numpy()
    print(f"Starting linear spatial regression process...")

    distmat = get_distmat("/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/gm_only_MNI152_1mm_desikan+aseg.nii.gz")
    print(f"Distance matrix calculated of size {distmat.shape}")

    y = np.array(cleaned_rois).reshape(-1,1)

    model = LinearRegression(fit_intercept=False)
    model.fit(X,y)
    r_squared = get_r_sq(X, y, model)
    adj_r_squared = get_adj_r_sq(X, y, model)

    pval,_ = get_regr_p_val_moran(X,y, output_dir = output_dir, atlas = "/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/gm_only_MNI152_1mm_desikan+aseg.nii.gz", distmat = distmat)
        
    beta_coeffs = model.coef_.tolist()
        
    coeff_row = {
        "DATA_KEY" : [nb_basename],
        f"D1" : [float(beta_coeffs[0][0])],
        f"D2" : [float(beta_coeffs[0][1])],
        f"DAT" : [float(beta_coeffs[0][2])],
        f"NET" : [float(beta_coeffs[0][3])],
        f"5HT1A" : [float(beta_coeffs[0][4])],
        f"5HT1B" : [float(beta_coeffs[0][5])],
        f"5HT2A" : [float(beta_coeffs[0][6])],
        f"5HT4" : [float(beta_coeffs[0][7])],
        f"5HT6" : [float(beta_coeffs[0][8])],
        f"5HTT" : [float(beta_coeffs[0][9])],
        f"a4b2" : [float(beta_coeffs[0][10])],
        f"M1" : [float(beta_coeffs[0][11])],
        f"vAChT" : [float(beta_coeffs[0][12])],
        f"NMDA" : [float(beta_coeffs[0][13])],
        f"mGluR5" : [float(beta_coeffs[0][14])],
        f"GABAA/BZ" : [float(beta_coeffs[0][15])],
        f"H3" : [float(beta_coeffs[0][16])],
        f"CB1" : [float(beta_coeffs[0][17])],
        f"MOR" : [float(beta_coeffs[0][18])]
    }
        
    coeff_df = pd.DataFrame(coeff_row)

    stats_row = {
        "DATA_KEY" : [nb_basename],
        f"R-Squared" : [float(r_squared[0])],
        f"Adjusted R-Squared" : [float(adj_r_squared[0])],
        f"p-value" : [float(pval)]
    }

    stats_df = pd.DataFrame(stats_row)

    print('Linear regression complete. Coefficinets and regression statistics are being saved...')

    os.makedirs('linreg_stats', exist_ok = True)

    coeff_df_path = os.path.join('linreg_stats', f"{nb_basename}_coeffs.csv")
    print(f"\nSaving coefficients to {coeff_df_path}")
    coeff_df.to_csv(coeff_df_path, index = False)

    stats_df_path = os.path.join('linreg_stats', f"{nb_basename}_stats.csv")
    print(f"\nSaving linear regression stats to {stats_df_path}")
    stats_df.to_csv(stats_df_path, index = False)

    class ResultsClass:
        def __init__(self, linreg_r_squared, linreg_adj_r_squared, linreg_pval):
            self.linreg_r_squared = linreg_r_squared
            self.linreg_adj_r_squared = linreg_adj_r_squared
            self.linreg_pval = linreg_pval

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib import font_manager
    cmap = np.genfromtxt('/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/colourmap.csv', delimiter=',')
    cmap_div = ListedColormap(cmap)
    plt.figure(figsize = (20,12))
    coeff_df_to_plot = coeff_df.drop(columns = 'DATA_KEY')
    sns.heatmap(coeff_df_to_plot,
                xticklabels = coeff_df_to_plot.columns.tolist(),
                vmin = -0.5,
                vmax = 0.5,
                cmap = cmap_div,
                annot = False)
    os.makedirs('plots', exist_ok = True)
    heatmap_path = os.path.join('plots', f'{nb_basename}_heatmap.png')
    plt.savefig(heatmap_path, dpi = 600)
    print(f'Heatmap of beta coefficient values saved to {heatmap_path}')
    linreg_results = ResultsClass(float(r_squared[0]), float(adj_r_squared[0]), float(pval))
    return linreg_results, coeff_df

def machine_learner(coeff_df, output_dir, input_file):
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import xgboost
    from xgboost import XGBClassifier
    import shap
    from shap import KernelExplainer
    seed = 42
    train_df = pd.read_csv('/mnt/c/Users/User/Downloads/neurobeta/neurobeta_standards/nb_train.csv', low_memory = False)
    y = train_df.copy()['DIAGNOSIS']
    X_raw = train_df.copy().drop(columns = 'DIAGNOSIS')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    xgb_balancer = float(y[y == 0].shape[0])/ float(y[y == 1].shape[0])
    print(xgb_balancer)

    ## Instantiate xgb with the balancer - mitigates class imbalance
    
    xgb = XGBClassifier(random_state = seed, scale_pos_weight = xgb_balancer)
    lr = LogisticRegression(random_state = seed, class_weight = 'balanced')

    xgb.fit(X_scaled, y)
    lr.fit(X_scaled, y)

    X_test_raw = coeff_df.drop(columns = 'DATA_KEY')
    X_test = scaler.transform(X_test_raw)
    y_pred = xgb.predict(X_test)
    y_predict_proba = xgb.predict_proba(X_test)

    diag_dict = {0 : 'Cognitively Normal', 1 : 'Alzheimer\'s Disease'}
    print(f'Results from XGB classifier. Likely diagnosis of your scan = {diag_dict[y_pred[0]]}. Probability of prediction certainty = {round(float(y_predict_proba[0][0]), 2) * 100 if diag_dict[y_pred[0]] == 0 else round(float(y_predict_proba[0][1]), 2) * 100}%')

    y_pred_lr = lr.predict(X_test)
    y_predict_proba_lr = lr.predict_proba(X_test)

    pred = int(y_pred_lr[0])
    proba = float(y_predict_proba_lr[0, pred]) * 100

    print(f"Results from LR classifier. Likely diagnosis of your scan = {diag_dict[pred]}. Probability of prediction certainty = {proba:.2f}%")
    explainer = shap.TreeExplainer(xgb, X_test)
    shap_values = explainer(X_test)
    shap_values.feature_names = X_test_raw.columns.tolist()

    shap.plots.waterfall(shap_values[0], max_display = 19, show = False)
    plt.savefig(f'plots/{nb_getbasename(input_file)}_shap_waterfall.png', dpi = 600)
    plt.close()


