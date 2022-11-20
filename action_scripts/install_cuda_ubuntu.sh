set -e
set -x
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install --setopt=obsoletes=0 -y \
    cuda-nvcc-11-2-11.2.152-1 \
    cuda-cudart-devel-11-2-11.2.152-1 \
    libcurand-devel-11-2-10.2.3.152-1 \
    libcudnn8-devel-8.1.1.33-1.cuda11.2 \
    libcublas-devel-11-2-11.4.1.1043-1
ln -s cuda-11.2 /usr/local/cuda