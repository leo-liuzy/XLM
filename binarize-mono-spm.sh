#!/bin/zsh


set -e

SPM_PATH="xlm-roberta-base"
OUTPATH="data/XLM_en_zh-Hans_ar_debug"

mkdir -p $OUTPATH

for lg in zh-Hans; do
  for split in train valid test; do
    python preprocess_spm.py $SPM_PATH $OUTPATH/$lg.$split
  done
done
