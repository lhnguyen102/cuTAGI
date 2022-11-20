set -e
set -x
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install --setopt=obsoletes=0 -y cuda-nvcc-11-2-11.2.152-1 cuda-cudart-devel-11-2-11.2.152-1
ln -s cuda-11.2 /usr/local/cuda
echo "PATH=/usr/local/cuda/bin:$PATH" >> $GITHUB_PATH
echo "LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> $GITHUB_ENV