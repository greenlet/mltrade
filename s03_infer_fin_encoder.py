from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import torch
import torch.utils.tensorboard as tb
from tqdm import trange

from mlt.data.ds_prices import DsPrices, BatchResType
from mlt.model.fin_transformer import FinEncoder
from mlt.train.fin_transformer import masked_mse_loss


class ArgsInfer(BaseModel):
    ds_file_path: Path = Field(
        None,
        required=False,
        description='Dataset CSV file path.',
        cli=('--ds-file-path',),
    )
    train_path: Path = Field(
        ...,
        required=True,
        description='Path to train directory.',
        cli=('--train-path',),
    )
    device: str = Field(
        'cpu',
        required=False,
        description='Device to run inference on. Can have values: "cpu", "cuda"',
        cli=('--device',),
    )
    days: int = Field(
        ...,
        required=True,
        description='Number of days in input. Last',
        cli=('--days',),
    )
    predict_horizons: list[int] = Field(
        ...,
        required=True,
        description='Prediction intervals. 1 - 5 min, 3 - 15 min, etc.',
        cli=('--predict-horizons',),
    )
    batch_size: Optional[int] = Field(
        None,
        required=False,
        description='Batch size.',
        cli=('--batch-size',),
    )


def main(args: ArgsInfer) -> int:
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
    inp_size = args.days * 105
    device = torch.device(args.device)
    print(f'Pytorch device: {device}')

    model = FinEncoder(
        inp_dim=inp_dim, out_dim=out_dim, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
        d_model=d_model, d_inner=d_inner, dropout=dropout_rate,
    )
    model = model.to(device)

    ds = DsPrices(
        fpath=args.ds_file_path, batch_size=args.batch_size, min_inp_size=inp_size,
        max_inp_size=inp_size, train_ratio=train_ratio,
    )

    checkpoint_fpath = args.train_path / 'best.pth'
    checkpoint = torch.load(checkpoint_fpath)
    print(f'Checkpoint keys: {list(checkpoint.keys())}')
    model.load_state_dict(checkpoint['model'])

    df = ds.df.copy()
    df['Train'] = True
    df.loc[ds.n_train:, 'Train'] = False
    timestamps = df['Timestamp'].to_numpy()
    timestamps = (timestamps - timestamps[0]) / ds.min_timestamp_diff
    timestamps_int = timestamps.astype(int)
    df.set_index(timestamps_int)
    n_total = len(df)

    def stock_hor_name(stock_name: str, hor: int) -> str:
        return f'{stock_name}Pred_{hor}'

    print('Generating stock prices prediction column names:')
    n_prices = len(ds.prices_names)
    strs = []
    for hor in args.predict_horizons:
        for i, stock_name in enumerate(ds.prices_names):
            col_name = stock_hor_name(stock_name, hor)
            df[col_name] = ''
            prefix = ' ' * 4 if i == 0 else ', '
            suffix = '' if i < n_prices - 1 else '\n'
            strs.append(f'{prefix}{col_name}{suffix}')
    print(''.join(strs))
    prices = df[ds.prices_names].to_numpy()

    model.eval()
    n_steps = n_total - inp_size
    max_hor = max(args.predict_horizons)
    pbar = trange(n_steps, desc=f'Eval', unit='batch')
    for step in pbar:
        inds = slice(step, step + inp_size)
        ps, ts = prices[inds], timestamps[inds]
        ps0 = ps[0]
        ps, ts = ps / ps0, ts - ts[0]
        batch = BatchResType(prices=ps[None, ...], timestamp=ts[None, ...])
        inp, tgt, div = batch.get_last_masked_tensors(args.predict_horizons)
        inp, tgt, div = inp.to(device), tgt.to(device), div.to(device)
        out, *_ = model(inp)
        ps_out: np.ndarray = out[:, -max_hor:].cpu().detach().numpy()
        for i_hor, hor in enumerate(args.predict_horizons):
            ts_ind = int(ts[-hor])
            ps_all_pred = ps_out[i_hor][-hor:] * ps0
            for i_stock, stock_name in enumerate(ds.prices_names):
                ps_stock_pred = ps_all_pred[:, i_stock]
                col_name = stock_hor_name(stock_name, hor)
                df.loc[ts_ind, col_name] = ';'.join(f'{pred:0.5f}' for pred in ps_stock_pred)
        # break
    pbar.close()

    out_fname = args.ds_file_path.with_suffix('').name
    out_fname = f'{out_fname}_{args.train_path.name}.csv'
    out_fpath = args.ds_file_path.parent / out_fname
    df.to_csv(out_fpath)

    return 0


if __name__ == '__main__':
    run_and_exit(ArgsInfer, main, 'Run financial transformer model inference.')

