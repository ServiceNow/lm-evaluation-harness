#!/bin/bash

# accelerate launch
MODEL_PATHS=()

for MODEL_PATH in "${MODEL_PATHS[@]}"

do
    MODEL_NAME="${MODEL_PATH##*/}"
    TARGET_DIRECTORY="/mnt/rishabh/lm-evaluation-harness/output/"
    FINAL_PATH="${TARGET_DIRECTORY}${MODEL_NAME}"

    echo "Running command for MODEL: $MODEL_NAME"

    lm_eval --model hf --model_args pretrained=$MODEL_PATH  --tasks xquad_ar,xquad_de,xquad_el,xquad_en,xquad_es,xquad_hi,xquad_ro,xquad_ru,xquad_th,xquad_tr,xquad_vi,xquad_zh --device cuda:0  --batch_size 4 --log_samples --output_path $FINAL_PATH --limit 100 --num_fewshot 3

    lm_eval --model hf --model_args pretrained=$MODEL_PATH --tasks tydiqa --device cuda:0  --batch_size 4 --log_samples --output_path $FINAL_PATH --limit 1000 --num_fewshot 3

    lm_eval --model hf --model_args pretrained=$MODEL_PATH  --tasks xnli,xcopa --device cuda:0  --batch_size 8 --log_samples --output_path $FINAL_PATH

    lm_eval --model hf --model_args pretrained=$MODEL_PATH  --tasks xlsum_mistral --device cuda:0  --batch_size 4 --log_samples --output_path $FINAL_PATH --limit 100

    lm_eval --model hf --model_args pretrained=$MODEL_PATH  --tasks mlqa_ar,mlqa_de,mlqa_en,mlqa_es,mlqa_hi,mlqa_vi,mlqa_zh --device cuda:0  --batch_size 4 --log_samples --output_path $FINAL_PATH --limit 100 --num_fewshot 3

    lm_eval --model hf --model_args pretrained=$MODEL_PATH  --tasks mgsm_cot_native --device cuda:0  --batch_size 8 --log_samples --output_path $FINAL_PATH --limit 100

done
