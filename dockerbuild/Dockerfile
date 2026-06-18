# Use Ubuntu 22.04.5 (ubuntu:22.04 typically points to the latest point release)
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 1. Install System Dependencies
# Includes dependencies for Python build, MATLAB Runtime, FSL, and SPM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    wget \
    git \
    pkg-config \
    libffi-dev \
    liblzma-dev \
    ca-certificates \
    unzip \
    bc \
    dc \
    file \
    libfontconfig1 \
    libfreetype6 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libice6 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    libxrandr2 \
    libxt6 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python 3.11.9 from Source
# Required to guarantee exactly 3.11.9 (PPAs often update to 3.11.10+)
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz && \
    tar -xf Python-3.11.9.tgz && \
    cd Python-3.11.9 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    ln -s /usr/local/bin/python3.11 /usr/local/bin/python3 && \
    ln -s /usr/local/bin/pip3.11 /usr/local/bin/pip3 && \
    cd .. && rm -rf Python-3.11.9*

# 3. Install MATLAB Runtime 2024b (R2024b)
RUN mkdir /opt/mcr_install && \
    cd /opt/mcr_install && \
    wget -q https://ssd.mathworks.com/supportfiles/downloads/R2024b/deployment_files/R2024b/installers/glnxa64/MATLAB_Runtime_R2024b_glnxa64.zip && \
    unzip -q MATLAB_Runtime_R2024b_glnxa64.zip && \
    ./install -mode silent -agreeToLicense yes && \
    cd / && rm -rf /opt/mcr_install

# Set MATLAB Runtime environment variables (Update v242 path if minor version differs)
ENV LD_LIBRARY_PATH=/usr/local/MATLAB/MATLAB_Runtime/R2024b/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2024b/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2024b/sys/os/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2024b/extern/bin/glnxa64:$LD_LIBRARY_PATH

# 4. Install SPM25 Standalone
# Note: Ensure the URL matches the latest SPM25 release assets on GitHub
RUN mkdir /opt/spm25 && \
    wget -q https://github.com/spm/spm/releases/download/25.01.02/spm_25.01.02_Linux_R2024b.zip -O /opt/spm25.zip && \
    unzip -q /opt/spm25.zip -d /opt/spm25 && \
    rm /opt/spm25.zip

# Create a shortcut for SPM
ENV PATH="/opt/spm25:$PATH"

# 5. Install FSL 6.0.7.22
# FSL 6.0.7+ uses a conda-based installer. We specify the version explicitly.
RUN wget -q https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py && \
    python3 fslinstaller.py -d /opt/fsl -V 6.0.7.22 && \
    rm fslinstaller.py

# Set FSL environment variables
ENV FSLDIR=/opt/fsl
ENV PATH=${FSLDIR}/share/fsl/bin:${PATH}
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV LD_LIBRARY_PATH=${FSLDIR}/lib:${LD_LIBRARY_PATH}

# 6. Setup Python Virtual Environment and Install Libraries
WORKDIR /app
COPY venv_nb_requirements.txt /app/venv_nb_requirements.txt
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install libraries (assuming venv_nb_requirements is a standard requirements file)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r venv_nb_requirements.txt || bash ./venv_nb_requirements.txt

# Final configuration
ENTRYPOINT ["/bin/bash"]
