#!/bin/bash

BASE_DIR="/home/kompiel/python_scripts/processing_files/asr-evaluate/eval_output/eg_complete"

VARIANTS=("whisperx" "whisper" "transformers" "whisper_timestamped" "crisper_whisper")

for variant in "${VARIANTS[@]}"; do

  OUT_DIR="${BASE_DIR}/large_eg_complete_${variant}"
  
  JSON_FILE="${OUT_DIR}/large_eg_complete_${variant}_seed42.json"
  LOG_FILE="${OUT_DIR}/large_eg_complete_${variant}_seed42.log"
  
  echo "Starte Auswertung für Variante: ${variant}"
  
  # python evaluate_dataset.py custom --seed=1337 --variant=${variant} --test_size=0.2 --output "${JSON_FILE}" 2>&1 | tee "${LOG_FILE}"

  python evaluate_dataset.py custom --seed=42 --variant=${variant} --test_size=0.5 --output "${JSON_FILE}" 2>&1 | tee "${LOG_FILE}"
  
  echo "Auswertung für ${variant} abgeschlossen"
  echo "------------------------"
done