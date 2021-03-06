JOB_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")"
REL_PATH=../../../

PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
srun \
--job-name="${JOB_NAME}"  \
--mpi=pmi2 -p $1 -n4 --gres=gpu:4 \
--ntasks-per-node=4 \
--cpus-per-task=5 \
python -u -m train \
--sh_name="$0" \
--cfg_name="cfg.yaml" \
--exp_dir_name="${EXP_DIR}" \

RESULT=$(tail "${EXP_DIR}"/log.txt -n 1)
echo ""
echo -e "\033[36mat ${PWD#}/${EXP_DIR}\033[0m"
echo -e "\033[36m${RESULT#*@}\033[0m"
