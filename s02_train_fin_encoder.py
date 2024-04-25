from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import torch
import torch.utils.tensorboard as tb
from tqdm import trange

from mlt.data.ds_prices import DsPrices
from mlt.model.fin_transformer import FinEncoder


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


def masked_mse_loss(out: torch.Tensor, tgt: torch.Tensor, div: torch.Tensor) -> torch.Tensor:
    mask = tgt > 0
    diff = torch.masked_select(out, mask) - torch.masked_select(tgt, mask)
    diff /= torch.masked_select(div, mask)
    return torch.mean(diff**2)


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
    min_inp_zeros = 2
    max_inp_zeros_rate = 0.6
    device = torch.device(args.device)
    print(f'Pytorch device: {device}')

    model = FinEncoder(
        inp_dim=inp_dim, out_dim=out_dim, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
        d_model=d_model, d_inner=d_inner, dropout=dropout_rate,
    )
    model = model.to(device)

    ds = DsPrices(
        fpath=args.ds_file_path, batch_size=args.batch_size, min_inp_size=min_inp_size,
        max_inp_size=max_inp_size, min_inp_zeros=min_inp_zeros, max_inp_zeros_rate=max_inp_zeros_rate,
        train_ratio=train_ratio,
    )

    train_subdir_name = gen_train_subdir_name()
    train_path = args.train_root_path / train_subdir_name
    train_path.mkdir(parents=True, exist_ok=True)
    print(f'Train path: {train_path}')
    tbsw = tb.SummaryWriter(log_dir=str(train_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        train_it = ds.get_train_it(args.train_epoch_steps, with_tensor=True)
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        model.train()
        train_loss = 0
        for step in pbar:
            batch = next(train_it)
            inp, tgt, div = batch.inp_t, batch.tgt_t, batch.div_t
            inp, tgt, div = inp.to(device), tgt.to(device), div.to(device)
            optimizer.zero_grad()
            out, *_ = model(inp)
            loss = masked_mse_loss(out, tgt, div)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)

        val_it = ds.get_val_it(args.val_epoch_steps, with_tensor=True)
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        model.eval()
        val_loss = 0
        for step in pbar:
            batch = next(val_it)
            inp, tgt, div = batch.inp_t, batch.tgt_t, batch.div_t
            inp, tgt, div = inp.to(device), tgt.to(device), div.to(device)
            out, *_ = model(inp)
            loss = masked_mse_loss(out, tgt, div)
            val_loss += loss.item()
        val_loss /= args.val_epoch_steps
        tbsw.add_scalar('Loss/Val', val_loss, epoch)

    return 0


if __name__ == '__main__':
    run_and_exit(ArgsTrain, main, 'Train financial transformer model.')



