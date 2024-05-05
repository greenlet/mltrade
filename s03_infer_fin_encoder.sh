#!/bin/zsh

code_path=$HOME/prog
data_path=$HOME/data
mltrade_root_path=$data_path/mltrade
ds_file_path=$mltrade_root_path/stocks_1.1.18-9.2.24_close.csv
train_root_path=$data_path/train_mltrade

train_subdir=20240427_204926
device=cpu
days=2
batch_size=5

train_path=$train_root_path/$train_subdir
mltrade_src_path=$code_path/mltrade
export PYTHONPATH=$PYTHONPATH:$mltrade_src_path

cd "$mltrade_src_path" || exit 1
python s03_infer_fin_encoder.py \
  --ds-file-path $ds_file_path \
  --train-path $train_path \
  --device $device \
  --days $days \
  --batch-size $batch_size

