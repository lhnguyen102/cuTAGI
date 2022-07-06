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
echo "argument $0"
docker run --rm \
            --$1 \
            -it \
            -e VAR1=$2 \
            -v $CONFIG_DIR:/usr/src/cutagi/cfg \
            -v $DATA_DIR:/usr/src/cutagi/data \
            -v $SAVED_PARAM_DIR:/usr/src/cutagi/saved_param \
            -v $SAVED_RESULTS_DIR:/usr/src/cutagi/saved_results \
            cutagi:0.1.2 
