DIR_NAME="${PWD##*/}"
REL_PATH=../../
CUR_PATH=$(pwd)
cd "${REL_PATH}"
PROJ_PATH=$(pwd)
cd "${CUR_PATH}"

PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
python -u -m cache 6 "/content/drive/MyDrive/datasets/UCRArchive_2018" "${1:-"None"}"
