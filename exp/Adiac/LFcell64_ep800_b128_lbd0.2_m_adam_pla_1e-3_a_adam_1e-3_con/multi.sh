REL_PATH=../../../

TIMES=$1
if [[ -z "$1" ]]; then
  TIMES=4
fi

MULTI_LINE_EXP=""
MULTI_LINE_RESULT=""
for i in $(seq 1 $TIMES)
do
  EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")"
  PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
  python -u -m train \
  --sh_name="$0" \
  --cfg_name="cfg.yaml" \
  --exp_dir_name="${EXP_DIR}" \

  CUR_RESULT=$(tail "${EXP_DIR}"/log.txt -n 1)
  MULTI_LINE_EXP="${MULTI_LINE_EXP}\n\033[36mat ${EXP_DIR}:\033[0m"
  MULTI_LINE_RESULT="${MULTI_LINE_RESULT}\n\033[36m${CUR_RESULT#*@}\033[0m\033[0m"
done

echo ""
echo -e "${MULTI_LINE_EXP}"
echo ""
echo -e "${MULTI_LINE_RESULT}"
