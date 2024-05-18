from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Optional

import numpy as np
import torch
from torch import nn

from mlt.data.ds_prices import BatchResType

DIRNAME_DT_PAT = '%Y%m%d_%H%M%S'
DIRNAME_RE_PAT = re.compile(r'(?P<dt>\d{8}_\d{6})')


def gen_dirname(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = datetime.now()
    return dt.strftime(DIRNAME_DT_PAT)


def parse_dirname(dirname: str) -> datetime:
    return datetime.strptime(dirname, DIRNAME_DT_PAT)


def find_last_subdir(dir_path: Path) -> Optional[str]:
    subdir_last, dt_last = None, None
    for dpath in dir_path.iterdir():
        if not dpath.is_dir():
            continue
        m = DIRNAME_RE_PAT.match(dpath.name)
        if not m:
            continue
        dt = parse_dirname(dpath.name)
        if dt_last is None or dt_last < dt:
            subdir_last, dt_last = dpath.name, dt
    return subdir_last


def masked_mse_loss(out: torch.Tensor, tgt: torch.Tensor, div: torch.Tensor) -> torch.Tensor:
    mask = tgt > 0
    diff = torch.masked_select(out, mask) - torch.masked_select(tgt, mask)
    # print(f'before: {diff}')
    diff /= torch.masked_select(div, mask)
    # print(f'div:    {torch.masked_select(div, mask)}')
    # print(f'after:  {diff}')
    return torch.mean(diff**2)


@dataclass
class FinMetric:
    horizon: int
    diff: float = -1
    diff_mean: float = 0


class FinMetricCalc:
    n_steps_total: int
    n_steps_calc: int
    last_zeros: list[int]
    model: nn.Module
    device: torch.device
    metrics: list[FinMetric]
    horizon_to_metrics: dict[int, FinMetric]
    calc_steps: np.ndarray
    calc_steps_set: set[int]

    def __init__(self, n_steps_total: int, n_steps_calc: int, last_zeros: list[int],
                 model: nn.Module, device: torch.device):
        self.n_steps_total = n_steps_total
        self.n_steps_calc = min(n_steps_calc, n_steps_total)
        self.last_zeros = last_zeros
        self.model = model
        self.device = device
        self.metrics = [FinMetric(horizon=nz) for nz in self.last_zeros]
        self.horizon_to_metrics = {met.horizon: met for met in self.metrics}
        if self.n_steps_calc == 1:
            # Calculate metrics on the last step. Otherwise, np.linspace will create [0] array
            calc_steps = np.array([self.n_steps_total - 1])
        else:
            calc_steps = np.linspace(0, self.n_steps_total - 1, self.n_steps_calc, endpoint=True, dtype=int)
        print(f'FinMetricCalc. calc_steps={calc_steps}')
        self.calc_steps = calc_steps
        self.calc_steps_set = set(self.calc_steps)

    def set_step(self, step: int, batch: BatchResType):
        if step not in self.calc_steps_set:
            return
        if step == self.calc_steps[0]:
            self.metrics = [FinMetric(horizon=nz) for nz in self.last_zeros]
        inp, tgt, div = batch.get_last_masked_tensors(self.last_zeros)
        inp, tgt, div = inp.to(self.device), tgt.to(self.device), div.to(self.device)
        training = self.model.training
        self.model.eval()
        out, *_ = self.model(inp)
        if training:
            self.model.train()
        for iz, met in enumerate(self.metrics):
            loss = masked_mse_loss(out[iz], tgt[iz], div[iz])
            met.diff = np.sqrt(loss.item())
            met.diff_mean += met.diff
        # print(self.metrics)
        self.horizon_to_metrics = {met.horizon: met for met in self.metrics}

        if step == self.calc_steps[-1]:
            for met in self.metrics:
                met.diff_mean /= self.n_steps_calc

