#!/bin/bash

# Переход в корень проекта (если скрипт запускается из scripts/)
cd "$(dirname "$0")/.." || exit

# Запуск теста
python test.py \
  --upscale 2 2 \
  --msb hdb \
  --lsb hd \
  --act-fn gelu \
  --n-filters 64 \
  --lut-dir ./luts \
  --test-dir ./data/test/