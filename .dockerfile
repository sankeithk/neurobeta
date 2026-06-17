FROM ubuntu:22.04

# ── System deps ──────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget curl unzip git \
    python3.11 python3.11-dev python3.11-pip \
    libgomp1 \
    dc bc \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3.11 1

# ── FSL ──────────────────────────────────────────────────────────────────────
# Neurodocker handles FSL cleanly — generate this block with:
# neurodocker generate docker --base-image ubuntu:22.04 --pkg-manager apt --fsl version=6.0.7.1
# Then paste the resulting FSL block here. Roughly:
ENV FSLDIR=/opt/fsl
ENV PATH="${FSLDIR}/bin:${PATH}"
ENV FSLOUTPUTTYPE=NIFTI_GZ
RUN wget -q https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py \
    && python fslinstaller.py -d /opt/fsl -V 6.0.7.1 --quiet \
    && rm fslinstaller.py

# ── MATLAB Runtime (MCR) ─────────────────────────────────────────────────────
# If issues with this, check Mathworks website for proper URL to MATLAB Runtime
ENV MCR_PATH=/opt/mcr/R2024b
RUN mkdir -p /opt/mcr && \
    wget -q https://ssd.mathworks.com/supportfiles/downloads/R2024b/Release/6/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2024b_Update_6_glnxa64.zip \
    -O /tmp/mcr.zip && \
    unzip -q /tmp/mcr.zip -d /tmp/mcr_installer && \
    /tmp/mcr_installer/install -destinationFolder /opt/mcr -agreeToLicense yes -mode silent && \
    rm -rf /tmp/mcr.zip /tmp/mcr_installer

# ── SPM Standalone ───────────────────────────────────────────────────────────
ENV SPM_ROOT=/opt/spm
ENV SPM_LAUNCHER=/opt/spm/run_spm25.sh
RUN mkdir -p /opt/spm && \
    wget -q https://github.com/spm/spm/releases/download/25.0/spm25_standalone_Linux.zip \
    -O /tmp/spm.zip && \
    unzip -q /tmp/spm.zip -d /opt/spm && \
    chmod +x /opt/spm/run_spm25.sh && \
    rm /tmp/spm.zip

# ── neurobeta ────────────────────────────────────────────────────────────────
WORKDIR /opt/neurobeta
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# ── Environment variables for neurobeta ──────────────────────────────────────
ENV SPM_LAUNCHER=/opt/spm/run_spm25.sh \
    MCR_PATH=/opt/mcr/R2024b \
    SPM_ROOT=/opt/spm

ENTRYPOINT ["python", "/opt/neurobeta/neurobeta.py"]