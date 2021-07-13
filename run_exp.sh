#!/bin/bash
python train.py \
    --exp_name xlm_en_zh-Hans \
    --dump_path ./dumped \
    --data_path data/XLM_en_zh-Hans_ar_debug \
    --model_name_or_path pretrained_models/xlm-roberta-base \
    --path_to_spm pretrained_models/xlm-roberta-base/sentencepiece.bpe.model \
    --lgs en,zh-Hans \
    --mlm_steps en,zh-Hans \
    --local_rank -1 \
    --batch_size 32 \
    --bptt 256 \
    --optimizer adam,lr=0.0001 \
    --epoch_size 300 \
    --max_epoch 10 \
    --validation_metrics _valid_mlm_ppl \
    --stopping_criterion _valid_mlm_ppl,25 \
    --lg_sampling_factor 1 \
    --sample_alpha 0.2 \
    --use_hg \
    --use_lang_emb False \
    --simcse_after_mlm False
#     --use_cpu
