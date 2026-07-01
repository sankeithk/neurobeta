# STAGE 1: Get MATLAB Runtime (MCR) from official source
# R2024b corresponds to MCR version v242
FROM containers.mathworks.com/matlab-runtime:r2024b AS mcr_source

# STAGE 2: Final Image
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 1. Install System Dependencies (Required for MCR, FSL, and SPM)
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

# 2. Install Python 3.11.9 (Built from source to ensure exact version)
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

# 3. Setup MATLAB Runtime 2024b
# We copy it from the MathWorks stage to avoid 404 download errors
COPY --from=mcr_source /opt/matlabruntime /opt/matlabruntime
ENV MCR_PATH=/opt/matlabruntime/v242
ENV LD_LIBRARY_PATH=${MCR_PATH}/runtime/glnxa64:${MCR_PATH}/bin/glnxa64:${MCR_PATH}/sys/os/glnxa64:${MCR_PATH}/extern/bin/glnxa64

# 4. Install SPM25 Standalone
# We download the 'Standalone' specific zip from GitHub
WORKDIR /opt
RUN wget -q https://github.com/spm/spm/releases/download/25.01.02/spm_standalone_25.01.02_Linux.zip && \
    unzip -q spm_standalone_25.01.02_Linux.zip -d spm25 && \
    rm spm_standalone_25.01.02_Linux.zip

# Create a helper script to run SPM Standalone easily
RUN echo '#!/bin/bash\n/opt/spm25/run_spm25.sh /opt/matlabruntime/v242 "$@"' > /usr/local/bin/spm25 && \
    chmod +x /usr/local/bin/spm25

# 5. Install FSL 6.0.7.22
RUN wget -q https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py && \
    python3 fslinstaller.py -d /opt/fsl -V 6.0.7.22 && \
    rm fslinstaller.py
ENV FSLDIR=/opt/fsl
ENV PATH=${FSLDIR}/share/fsl/bin:${PATH}
ENV FSLOUTPUTTYPE=NIFTI_GZ
# Add FSL libs to LD_LIBRARY_PATH (appending to MCR paths)
ENV LD_LIBRARY_PATH=${FSLDIR}/lib:${LD_LIBRARY_PATH}

# 6. Setup Python Virtual Environment and Install Libraries
WORKDIR /app
COPY venv_nb_requirements.txt /app/venv_nb_requirements.txt

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Run the installation logic for your requirements
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    if [ -f venv_nb_requirements.txt ]; then \
    if grep -q "pip install" venv_nb_requirements; then \
    chmod +x venv_nb_requirements.txt && ./venv_nb_requirements.txt; \
    else \
    pip install --no-cache-dir -r venv_nb_requirements.txt; \
    fi \
    fi

# Final Cleanup and Entrypoint
WORKDIR /data
ENTRYPOINT ["/bin/bash"]