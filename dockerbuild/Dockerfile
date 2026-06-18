# Use Ubuntu 22.04
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev curl wget git pkg-config libffi-dev liblzma-dev \
    ca-certificates unzip bc dc file libfontconfig1 libfreetype6 \
    libgl1-mesa-dev libglib2.0-0 libice6 libxcursor1 libxft2 \
    libxinerama1 libxrandr2 libxt6 sudo \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python 3.11.9 from Source
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
# Added a User-Agent to prevent the Server Error (Exit Code 8)
RUN mkdir /opt/mcr_install && \
    cd /opt/mcr_install && \
    wget --user-agent="Mozilla/5.0" https://ssd.mathworks.com/supportfiles/downloads/R2024b/deployment_files/R2024b/installers/glnxa64/MATLAB_Runtime_R2024b_glnxa64.zip && \
    unzip -q MATLAB_Runtime_R2024b_glnxa64.zip && \
    ./install -mode silent -agreeToLicense yes && \
    cd / && rm -rf /opt/mcr_install

# Set MATLAB Runtime environment variables (Fixed the UndefinedVar warning)
ENV LD_LIBRARY_PATH=/usr/local/MATLAB/MATLAB_Runtime/R2024b/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2024b/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2024b/sys/os/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2024b/extern/bin/glnxa64

# 4. Install SPM25 Standalone
RUN mkdir /opt/spm25 && \
    wget --user-agent="Mozilla/5.0" https://github.com/spm/spm/releases/download/25.01.02/spm_25.01.02_Linux_R2024b.zip -O /opt/spm25.zip && \
    unzip -q /opt/spm25.zip -d /opt/spm25 && \
    rm /opt/spm25.zip
ENV PATH="/opt/spm25:$PATH"

# 5. Install FSL 6.0.7.22
# Added -y to ensure the installer doesn't wait for your input
RUN wget -q https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py && \
    python3 fslinstaller.py -d /opt/fsl -V 6.0.7.22 -y && \
    rm fslinstaller.py

# Set FSL environment variables
ENV FSLDIR=/opt/fsl
ENV PATH=${FSLDIR}/share/fsl/bin:${PATH}
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV LD_LIBRARY_PATH=${FSLDIR}/lib:${LD_LIBRARY_PATH}

# 6. Setup Python Virtual Environment and Install Libraries
WORKDIR /app

# IMPORTANT: Ensure your local file is named EXACTLY 'venv_nb_requirements.txt'
# Or change the line below to match your filename.
COPY venv_nb_requirements.txt /app/venv_nb_requirements.txt

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/venv_nb_requirements.txt

ENTRYPOINT ["/bin/bash"]