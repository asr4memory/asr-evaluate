#!/bin/bash

BASE_DIR="/Users/pkompiel/python_scripts/asr4memory/processing_files/asr-evaluate/_variant_eval_output/eg_fzh-wde_combined_dataset_v2"

VARIANTS=("whisperx" "whisper" "transformers" "whisper_timestamped" "crisper_whisper" "wav2vec" "whisper_mlx")

for variant in "${VARIANTS[@]}"; do

  OUT_DIR="${BASE_DIR}/large_eg_complete_${variant}"

  mkdir -p "${OUT_DIR}"
  
  JSON_FILE="${OUT_DIR}/large_eg_complete_${variant}_seed42.json"
  LOG_FILE="${OUT_DIR}/large_eg_complete_${variant}_seed42.log"
  
  echo "Starte Auswertung für Variante: ${variant}"
  
  python -u evaluate_dataset.py custom --seed=1337 --variant=${variant} --test_size=0.9 --output "${JSON_FILE}" 2>&1 | tee "${LOG_FILE}"

  # python evaluate_dataset.py custom --seed=42 --variant=${variant} --test_size=0.5 --output "${JSON_FILE}" 2>&1 | tee "${LOG_FILE}"
  
  echo "Auswertung für ${variant} abgeschlossen"
  echo "------------------------"
done