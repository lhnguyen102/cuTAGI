 #!/bin/bash


#####################
## DIRECTORIES
#####################
script="$0"
FOLDER="$(pwd)/$(dirname $script)"

source $FOLDER/utils.sh
PROJECT_ROOT="$(abspath $FOLDER/..)"
echo "project root folder $PROJECT_ROOT"

DATA_DIR=$PROJECT_ROOT/data
echo "Using data in $DATA_DIR"

CONFIG_DIR=$PROJECT_ROOT/cfg
echo "Using data in $CONFIG_DIR"

SAVED_PARAM_DIR=$PROJECT_ROOT/saved_param
echo "Using data in $SAVED_PARAM_DIR"

SAVED_RESULTS_DIR=$PROJECT_ROOT/saved_results
echo "Using data in $SAVED_RESULTS_DIR"

#####################
## ARGUMENT PASSING
#####################
unset -v CFG VERSION DEVICE
# TEMP=$(getopt -n "$0" -a -l "cfg:,version:,device:" -- -- "$@")
# [ $? -eq 0 ] || exit
# eval set --  "$TEMP"
# while [ $# -gt 0 ]
# do
#     case "$1" in
#         --cfg) CFG="$2"; shift;;
#         --version) VERSION="$2"; shift;;
#         --device) DEVICE="$2"; shift;;
#         --) shift;;
#     esac
#     shift;
# done

while getopts "c:v:d:" flag;
do
    case "${flag}" in
        c) CFG=${OPTARG};;
        v) VERSION=${OPTARG};;
        d) DEVICE=${OPTARG};;
    esac
done

if [ -z "${CFG+set}" ]; then
    echo "ERROR: User input file is not provided"
    exit 1;
fi
if [ -z "${VERSION+set}" ]; then
    VERSION="latest"
fi
if [ -z "${DEVICE+set}" ]; then
    DEVICE="cpu" 
fi

if [ ${DEVICE} != "cuda" ]; then
    docker run --rm \
                -it \
                -e VAR1=${CFG} \
                -v $CONFIG_DIR:/usr/src/cutagi/cfg \
                -v $DATA_DIR:/usr/src/cutagi/data \
                -v $SAVED_PARAM_DIR:/usr/src/cutagi/saved_param \
                -v $SAVED_RESULTS_DIR:/usr/src/cutagi/saved_results \
                cutagi:${VERSION}
else 
    docker run --rm \
                    -it \
                    --gpus=all \
                    -e VAR1=${CFG} \
                    -v $CONFIG_DIR:/usr/src/cutagi/cfg \
                    -v $DATA_DIR:/usr/src/cutagi/data \
                    -v $SAVED_PARAM_DIR:/usr/src/cutagi/saved_param \
                    -v $SAVED_RESULTS_DIR:/usr/src/cutagi/saved_results \
                    cutagi:${VERSION}
fi
