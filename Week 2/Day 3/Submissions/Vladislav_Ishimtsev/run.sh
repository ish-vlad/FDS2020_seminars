#!/usr/bin/env bash

set -e

__usage="
Usage: $0 [-d data_dir] [-n n_jobs] [-v]

  -d: 	Directory with data
  -n: 	Number of jobs
  -v: 	Verbose. Do we need to activate all prints?
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
VERBOSE=false
DATA_DIR=./data
NUMBER_OF_JOBS=20
while getopts "d:n:v" opt
do
    case ${opt} in
        d) DATA_DIR=$OPTARG;;
        n) NUMBER_OF_JOBS=$OPTARG;;
        v) VERBOSE=true;;
        *) usage; exit 1 ;;
    esac
done

if [ "${VERBOSE}" = true ]; then
    set -x
    VERBOSE_ARG="-v"
fi

# Run
python main.py \\
        -d ${DATA_DIR} \\
        -n ${NUMBER_OF_JOBS} \\
        ${VERBOSE_ARG}
