set -e
set -x 
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum --enablerepo=epel -y install cuda-11-8

CUDA_PATH=/usr/local/cuda.11.8
echo "CUDA_PATH=${CUDA_PATH}"
export CUDA_PATH=${CUDA_PATH}

# Quick test. @temp
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib:$LD_LIBRARY_PATH"
nvcc -V

# If executed on github actions, make the appropriate echo statements to update the environment
if [[ $GITHUB_ACTIONS ]]; then
    # Set paths for subsequent steps, using ${CUDA_PATH}
    echo "Adding CUDA to CUDA_PATH, PATH and LD_LIBRARY_PATH"
    echo "CUDA_PATH=${CUDA_PATH}" >> $GITHUB_ENV
    echo "${CUDA_PATH}/bin" >> $GITHUB_PATH
    echo "LD_LIBRARY_PATH=${CUDA_PATH}/lib:${LD_LIBRARY_PATH}" >> $GITHUB_ENV
fi