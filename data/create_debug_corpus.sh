source_dir="XLM_pilot_run_21Langs"
# debug_dir="XLM_pilot_run_21Langs_debug"
debug_dir="${source_dir}_debug"

mkdir -p ${debug_dir}

cp ${source_dir}/*.test ${debug_dir}/
cp ${source_dir}/*.valid ${debug_dir}/

N_TRAIN=500000
for f in ${source_dir}/*.train; do
    LG="${f%.train}"  # trim extension
    LG="${LG#${source_dir}/}" # trim path to source_dir
    echo "Processing ${LG}"
    head -n $N_TRAIN $f > ${debug_dir}/${LG}.train
done
