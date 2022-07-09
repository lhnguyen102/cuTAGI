#!/bin/bash
#####################
## ARGUMENT PASSING
#####################
unset -v VERSION DEVICE
# TEMP=$(getopt -n "$0" -a -l "version:,device:" -- -- "$@")
# [ $? -eq 0 ] || exit
# eval set --  "$TEMP"
# while [ $# -gt 0 ]
# do
#     case "$1" in
#         --version) VERSION="$2"; shift;;
#         --device) DEVICE="$2"; shift;;
#         --) shift;;
#     esac
#     shift;
# done

###########################
## pass flags
while getopts "v:d:" flag;
do
    case "${flag}" in
        v) VERSION=${OPTARG};;
        d) DEVICE=${OPTARG};;
    esac
done

if [ -z "${VERSION+set}" ]; then
    VERSION="latest"
fi
if [ -z "${DEVICE+set}" ]; then
    DEVICE="cpu" 
fi

#####################
## DOCKER RUN
#####################
echo "Build version: ${VERSION}"
if [ ${DEVICE} == "cpu" ]; then
    echo "Build docker images with CPU..."
    docker build -f cpu.dockerfile . -t cutagi:${VERSION}
elif  [ ${DEVICE} == "cuda" ]; then
    echo "Build docker images with CUDA..."
    docker build . -t cutagi:${VERSION}
else
    echo "Device is not available"
fi
