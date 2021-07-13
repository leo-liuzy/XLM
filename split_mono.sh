source_dir="data/XLM_pilot_run_21Langs"
split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NTRAIN=$((NLINES - 10000));
    NVAL=$((NTRAIN + 5000));
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN             > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NVAL | tail -5000  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -5000                > $4;
}

echo "*** Split into train / valid / test ***"

for f in ${source_dir}/*.all; do
    LG="${f%.all}"  # trim extension
    LG="${LG#${source_dir}/}" # trim path to source_dir
    echo "Processing ${LG}"
    split_data ${source_dir}/${LG}.all ${source_dir}/${LG}.train ${source_dir}/${LG}.valid ${source_dir}/${LG}.test
done
