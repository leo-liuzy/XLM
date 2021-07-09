#!/bin/zsh


set -e

pair=ar-en
SPM_PATH="xlm-roberta-base"
OUTPATH="data/XLM_en_zh-Hans_ar_debug"

mkdir -p $OUTPATH

for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    python preprocess_spm.py $SPM_PATH $OUTPATH/$pair.$lg.$split
  done
done
