#!/bin/zsh

code_path=$HOME/prog
data_path=$HOME/data
mltrade_root_path=$data_path/mltrade
ds_file_path=$mltrade_root_path/stocks_1.1.18-9.2.24_close.csv
train_root_path=$data_path/train_mltrade

device=cpu
epochs=50
train_epoch_steps=20
val_epoch_steps=20
#epochs=3
#train_epoch_steps=10
#val_epoch_steps=10
batch_size=5
learning_rate=0.001

mltrade_src_path=$code_path/mltrade
export PYTHONPATH=$PYTHONPATH:$mltrade_src_path

cd "$mltrade_src_path" || exit 1
python s02_train_fin_encoder.py \
  --ds-file-path $ds_file_path \
  --train-root-path $train_root_path \
  --batch-size $batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps


