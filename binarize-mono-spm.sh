#!/bin/bash


set -e

SPM_PATH="pretrained_models/xlm-roberta-base"
source_dir="data/XLM_en_zh-Hans_ar_debug"

mkdir -p $source_dir

for f in ${source_dir}/*.all; do
    LG="${f%.all}"  # trim extension
    LG="${LG#${source_dir}/}" # trim path to source_dir
    echo "Processing ${LG}"
    for split in train valid test; do
        python preprocess_spm.py $SPM_PATH ${source_dir}/$LG.$split # > /dev/null 2>&1
    done
done
