"""
   Copyright 2026 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Detects a curve that a continuous device does not show in its history. An autoencoder reconstructs the smoothed last window of minutes, and an error above a threshold learned on held-back history is a hit.
"""

import dataclasses
import datetime
import json
import typing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import preprocessing
from ..model import WINDOW

LATENT_DIMS = 32
# One cause stays in the last window for its whole length and would be reported once per minute otherwise.
CURVE_LOCKOUT = datetime.timedelta(minutes=WINDOW)
DEBOUNCE = datetime.timedelta(minutes=30)
MIN_CURVE_DAYS = 3
EPOCHS = 60
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
# The newest fifth of the history is held back from the training. The errors on the training windows say
# little about the errors on unseen data (the longer the training, the further they fall below), so the
# threshold is a quantile of the errors on the held-back windows.
HOLDOUT_FRACTION = 0.2
THRESHOLD_QUANTILE = 0.999
MIN_HOLDOUT_WINDOWS = 100
TORCH_THREADS = 2


class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, stride=3)  # (205-7)/3+1 = 67
        self.conv2 = nn.Conv1d(16, 32, 7, stride=3)  # (67-7)/3+1 = 21
        self.fc1 = nn.Linear(672, latent_dims)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        x = x.reshape(-1, 1, WINDOW)
        x = F.relu(self.dropout(self.conv1(x)))
        x = F.relu(self.dropout(self.conv2(x)))
        return self.fc1(x.reshape(-1, 672))


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.fc1 = nn.Linear(latent_dims, 672)
        self.convt1 = nn.ConvTranspose1d(32, 16, kernel_size=7, stride=3)
        self.convt2 = nn.ConvTranspose1d(16, 1, kernel_size=7, stride=3)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, z):
        z = F.relu(self.dropout(self.fc1(z)))
        z = F.relu(self.convt1(z.reshape(-1, 32, 21)))
        return self.convt2(z).reshape(-1, WINDOW)


class Autoencoder(nn.Module):
    """(B, 205) or (B, 1, 205) in, (B, 205) out."""

    def __init__(self, latent_dims: int = LATENT_DIMS):
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        return self.decoder(self.encoder(x))


@dataclasses.dataclass
class ContCurveModel:
    state_dict: typing.Dict[str, torch.Tensor]
    training_max: float
    error_threshold: float
    _module: typing.Optional[nn.Module] = dataclasses.field(default=None, init=False, repr=False, compare=False)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_module"] = None
        return state

    def _net(self) -> nn.Module:
        if self._module is None:
            net = Autoencoder()
            net.load_state_dict(self.state_dict)
            net.eval()
            self._module = net
        return self._module

    def reconstruct(self, window: np.ndarray) -> typing.Tuple[np.ndarray, float]:
        """
        window: 205 smoothed minute values in the original unit. Returns the reconstruction in the original unit
        and the mean absolute error over the 205 normalised points.
        """
        window = np.asarray(window, dtype=float).reshape(-1)
        if window.shape[0] != WINDOW:
            raise ValueError(f"window needs {WINDOW} values, got {window.shape[0]}")
        normalised = window / self.training_max
        with torch.no_grad():
            out = self._net()(torch.as_tensor(normalised, dtype=torch.float32)).numpy().reshape(-1).astype(float)
        return out * self.training_max, float(np.abs(out - normalised).sum() / WINDOW)


def _train_autoencoder(data: np.ndarray, epochs: int) -> Autoencoder:
    """data: (n, 205) normalised windows."""
    torch.set_num_threads(TORCH_THREADS)
    torch.manual_seed(0)
    net = Autoencoder()
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    x = torch.as_tensor(data, dtype=torch.float32)
    generator = torch.Generator().manual_seed(0)
    net.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=generator).split(BATCH_SIZE):
            batch = x[idx]
            opt.zero_grad()
            loss = ((batch - net(batch)) ** 2).sum()
            loss.backward()
            opt.step()
    net.eval()
    return net


def _errors(net: Autoencoder, data: np.ndarray) -> np.ndarray:
    """Mean absolute reconstruction error of every window, as the runtime computes it."""
    x = torch.as_tensor(data, dtype=torch.float32)
    with torch.no_grad():
        return torch.cat([(net(b) - b).abs().sum(dim=1) / data.shape[1] for b in x.split(256)]).numpy().astype(float)


def _cont_curve(series: pd.Series, epochs: int, log: typing.Callable[[str], None]) -> typing.Optional[ContCurveModel]:
    minutes = preprocessing.to_minutes(series)
    if len(minutes) < MIN_CURVE_DAYS * 1440:
        return None
    training_max = float(series.max())
    if not np.isfinite(training_max) or training_max <= 0:
        return None
    smoothed = preprocessing.smooth(minutes / training_max).dropna().to_numpy()
    cut = int(len(smoothed) * (1 - HOLDOUT_FRACTION))
    # The windows of the two parts are cut separately, so no holdout window holds a minute the network saw.
    train_data = preprocessing.windows(smoothed[:cut], step=10)
    # Step 1 as the runtime checks every minute; every window of the day has to stay below the threshold.
    holdout_data = preprocessing.windows(smoothed[cut:], step=1)
    if len(train_data) == 0:
        return None
    if len(holdout_data) < MIN_HOLDOUT_WINDOWS:
        log(f"no curve model, {len(holdout_data)} holdout windows of {MIN_HOLDOUT_WINDOWS} needed")
        return None
    net = _train_autoencoder(train_data, epochs)
    return ContCurveModel(
        state_dict={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
        training_max=training_max,
        error_threshold=float(np.quantile(_errors(net, holdout_data), THRESHOLD_QUANTILE)))


def check(curve: ContCurveModel, st, key, ts, value) -> typing.Optional[dict]:
    st.push(ts, value)
    minute = ts.replace(second=0, microsecond=0)
    if st.last_check_minute is not None and minute <= st.last_check_minute:
        return None
    st.last_check_minute = minute
    if st.curve_locked_until is not None and ts < st.curve_locked_until:
        return None  # no report and no hit for the debounce while the last report is still in the window
    # The smoothing is centred and leaves the edges empty, so the last window ends about 15 minutes before now.
    smoothed = preprocessing.smooth(preprocessing.to_minutes(preprocessing.to_series(st.buffer))).dropna()
    if len(smoothed) < WINDOW:
        return None
    window = smoothed.iloc[-WINDOW:]
    reconstruction, error = curve.reconstruct(window.to_numpy())
    if not error > curve.error_threshold:
        return None
    previous, st.last_hit = st.last_hit, ts
    if previous is None or ts - previous > DEBOUNCE:
        return None
    st.curve_locked_until = ts + CURVE_LOCKOUT
    millis = window.index.as_unit("ns").asi8 // 10**6
    curves = [[int(m), float(o), float(r)] for m, o, r in zip(millis, window.to_numpy(), reconstruction)]
    return {"type": "curve_anomaly", "sub_type": "continous_device", "value": preprocessing.iso_utc(window.index[0]),
            "start_time": preprocessing.iso_utc(window.index[0]), "end_time": preprocessing.iso_utc(window.index[-1]),
            "original_reconstructed_curves": json.dumps(curves), "device_id": key}
