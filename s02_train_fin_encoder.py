import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import torch
import torch.utils.tensorboard as tb
from tqdm import trange

from mlt.data.ds_prices import DsPrices, MaskGenCfg
from mlt.model.fin_transformer import FinEncoder
from mlt.train.fin_transformer import masked_mse_loss, FinMetricCalc


class ArgsTrain(BaseModel):
    ds_file_path: Path = Field(
        None,
        required=False,
        description='Dataset CSV file path.',
        cli=('--ds-file-path',),
    )
    train_root_path: Path = Field(
        ...,
        required=True,
        description='Path to train root directory. New train processes will create subdirectories in it.',
        cli=('--train-root-path',),
    )
    batch_size: Optional[int] = Field(
        None,
        required=False,
        description='Batch size.',
        cli=('--batch-size',),
    )
    device: str = Field(
        'cpu',
        required=False,
        description='Device to run training on. Can have values: "cpu", "gpu"',
        cli=('--device',)
    )
    epochs: Optional[int] = Field(
        None,
        required=False,
        description='Number of training epochs.',
        cli=('--epochs',),
    )
    learning_rate: float = Field(
        0.001,
        required=False,
        description='Initial learning rate of the training process.',
        cli=('--learning-rate',)
    )
    train_epoch_steps: Optional[int] = Field(
        None,
        required=False,
        description='Number of training steps per epoch.',
        cli=('--train-epoch-steps',),
    )
    val_epoch_steps: Optional[int] = Field(
        None,
        required=False,
        description='Number of validation steps per epoch.',
        cli=('--val-epoch-steps',),
    )


def gen_train_subdir_name() -> str:
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    return dt_str


def main(args: ArgsTrain) -> int:
    print(args)

    inp_dim = 7
    out_dim = 6
    d_model = 128
    d_inner = d_model * 4
    n_layers = 4
    n_head = 4
    d_k = 32
    d_v = 32
    dropout_rate = 0.1
    train_ratio = 0.9
    min_inp_size = 20 * 105 # 105 5-min intervals in a trading day (10:00 - 18:45)
    max_inp_size = 90 * 105
    min_inp_size = 5 * 105
    max_inp_size = 20 * 105
    device = torch.device(args.device)
    print(f'Pytorch device: {device}')

    model = FinEncoder(
        inp_dim=inp_dim, out_dim=out_dim, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
        d_model=d_model, d_inner=d_inner, dropout=dropout_rate,
    )
    model = model.to(device)

    ds = DsPrices(
        fpath=args.ds_file_path, batch_size=args.batch_size, min_inp_size=min_inp_size,
        max_inp_size=max_inp_size, train_ratio=train_ratio,
    )

    train_subdir_name = gen_train_subdir_name()
    train_path = args.train_root_path / train_subdir_name
    train_path.mkdir(parents=True, exist_ok=True)
    print(f'Train path: {train_path}')
    tbsw = tb.SummaryWriter(log_dir=str(train_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    val_loss_min = None
    last_checkpoint_path, best_checkpoint_path = train_path / 'last.pth', train_path / 'best.pth'
    last_zeros = [1, 3, 5, 10]
    cfg_mask = MaskGenCfg(mid_min_num=2, mid_max_ratio=0.6, last_min_num=1, last_max_num=10)
    for epoch in range(args.epochs):
        train_it = ds.get_train_it(args.train_epoch_steps, cfgm=cfg_mask)
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        model.train()
        train_loss = 0
        train_metric = FinMetricCalc(args.train_epoch_steps, 10, last_zeros, model, device)
        for step in pbar:
            batch = next(train_it)
            inp, tgt, div = batch.get_masked_tensors()
            inp, tgt, div = inp.to(device), tgt.to(device), div.to(device)
            optimizer.zero_grad()
            out, *_ = model(inp)
            loss = masked_mse_loss(out, tgt, div)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_metric.set_step(step, batch)
            met_str = f'diff@1: {train_metric.horizon_to_metrics[1].diff:.3f}'

            pbar.set_postfix_str(f'Train. loss: {loss.item():.6f}. {met_str}')
        pbar.close()
        train_loss /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)
        for met in train_metric.metrics:
            tbsw.add_scalar(f'Diff@{met.horizon}/Train', met.diff_mean, epoch)

        val_it = ds.get_val_it(args.val_epoch_steps, cfgm=cfg_mask)
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        model.eval()
        val_loss = 0
        val_metric = FinMetricCalc(args.val_epoch_steps, 10, last_zeros, model, device)
        for step in pbar:
            batch = next(val_it)
            inp, tgt, div = batch.get_masked_tensors()
            inp, tgt, div = inp.to(device), tgt.to(device), div.to(device)
            out, *_ = model(inp)
            loss = masked_mse_loss(out, tgt, div)
            val_loss += loss.item()

            val_metric.set_step(step, batch)
            met_str = f'diff@1: {val_metric.horizon_to_metrics[1].diff:.3f}'

            pbar.set_postfix_str(f'Val. loss: {loss.item():.6f}. {met_str}')
        pbar.close()
        val_loss /= args.val_epoch_steps
        tbsw.add_scalar('Loss/Val', val_loss, epoch)
        for met in val_metric.metrics:
            tbsw.add_scalar(f'Diff@{met.horizon}/Val', met.diff_mean, epoch)

        print(f'Train loss: {train_loss:.6f}. Val loss: {val_loss:.6f}')
        best = False
        if val_loss_min is None or val_loss < val_loss_min:
            val_loss_str = f'{val_loss_min}' if val_loss_min is None else f'{val_loss_min:.6f}'
            print(f'Val min loss change: {val_loss_str} --> {val_loss:.6f}')
            val_loss_min = val_loss
            best = True

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'last_epoch': epoch,
            'val_loss_min': val_loss_min,
        }
        print(f'Saving checkpoint to {last_checkpoint_path}')
        torch.save(checkpoint, last_checkpoint_path)

        if best:
            print(f'New val loss minimum: {val_loss_min:.6f}. Saving checkpoint to {best_checkpoint_path}')
            shutil.copyfile(last_checkpoint_path, best_checkpoint_path)

    return 0


if __name__ == '__main__':
    run_and_exit(ArgsTrain, main, 'Train financial transformer model.')



