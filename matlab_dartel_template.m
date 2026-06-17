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