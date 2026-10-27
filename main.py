"""
   Copyright 2022 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import dotenv
dotenv.load_dotenv()

import datetime
import math
import threading
import time
import typing
import zoneinfo

import ray

from operator_lib import util
from operator_lib.util import MLOperator, Config, Inputs, Outputs, logger
from operator_lib.util import clock
from operator_lib.util.helpers import provide_historic_data

from algo import live
from algo.detectors import continuous_curve, extreme_value, load_curve, receive_time, schema
from algo.detectors.continuous_curve import ContCurveModel
from algo.detectors.load_curve import LoadCurveModel
from algo.model import SCHEMA_VERSION, AnomalyModel
from algo.training import train_anomaly_model

__all__ = ("Operator", "CustomConfig", "OperatorInputs", "OperatorOutputs")

HISTORY_DAYS = 50
# A requested history that cannot be read, as when the Ray cluster has no CPU free, is tried again this
# often, this far apart, before the day counts as trained on what could be read.
HISTORY_RETRY_S = 600
HISTORY_RETRIES = 6
# A message older than this (wall clock minus Kafka time) means the operator is catching up on a backlog.
LAG_LIMIT_S = 120
WATCHDOG_INTERVAL_S = 5
# After a training the watchdog stays silent only this long for a message that does not come.
PAUSE_GRACE_S = 600
BERLIN = zoneinfo.ZoneInfo("Europe/Berlin")
UTC = datetime.timezone.utc


def _parse_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    # A designer node hands over a config value it never set as an empty string.
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in ("true", "1")


class CustomConfig(Config):
    # Redeclared so the manifest lists it, with the library's default.
    logger_level: str = "warning"

    check_data_anomalies: bool = False
    check_data_extreme_outlier: bool = True
    check_data_schema: bool = True
    check_receive_time_outlier: bool = True

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        for name in ("check_data_anomalies", "check_data_extreme_outlier", "check_data_schema",
                     "check_receive_time_outlier"):
            setattr(self, name, _parse_bool(getattr(self, name), getattr(type(self), name)))


class OperatorInputs(Inputs):
    value: float
    original_input_ids: str


class OperatorOutputs(Outputs):
    type: str
    sub_type: str
    value: str
    initial_phase: str
    mean: float
    threshold: float
    device_id: str
    start_time: str
    end_time: str
    original_reconstructed_curves: str


_unwrap_warned = False


def _unwrap(model) -> typing.Optional[AnomalyModel]:
    """The model behind what mlflow loaded; one of another schema version counts as none."""
    global _unwrap_warned
    if model is None:
        return None
    if hasattr(model, "unwrap_python_model"):
        model = model.unwrap_python_model()
    if not isinstance(model, AnomalyModel) or model.schema_version != SCHEMA_VERSION:
        if not _unwrap_warned:
            _unwrap_warned = True
            logger.warning(f"ignoring a model of unexpected type or schema version: {type(model).__name__}")
        return None
    return model


def _topic_meta() -> typing.List[dict]:
    topics = util.OperatorConfig(util.load_operator_config_json(util.DeploymentConfig())).inputTopics
    return [{"filterType": t.filterType, "filterValue": t.filterValue} for t in topics]


def _aware(t: datetime.datetime) -> datetime.datetime:
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def _berlin_date(t: datetime.datetime) -> datetime.date:
    return _aware(t).astimezone(BERLIN).date()


class Operator(MLOperator):
    configType = CustomConfig
    inputs = OperatorInputs
    outputs = OperatorOutputs

    def __init__(self):
        # Guards the device state the watchdog thread shares with infer(), and the flags below.
        self._lock = threading.Lock()
        self._live: typing.Dict[str, live.DeviceState] = {}
        self._paused = False  # training or its aftermath: no missing-message reports
        self._pause_until: typing.Optional[float] = None  # monotonic end of the pause once the training is over
        self._catching_up = False  # the last message was old: a backlog, not an outage
        self._retry_at: typing.Optional[float] = None  # monotonic time of the next training after an unreadable history
        self._history_retries = 0
        self._stop_event = threading.Event()
        self._watchdog: typing.Optional[threading.Thread] = None

    # lifecycle

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        # After the first training, and the producer exists.
        self._watchdog = threading.Thread(target=self._watchdog_loop, name="outage-watchdog", daemon=True)
        self._watchdog.start()

    def stop(self):
        self._stop_event.set()
        if self._watchdog is not None and self._watchdog is not threading.current_thread():
            self._watchdog.join(timeout=WATCHDOG_INTERVAL_S + 5)
        super().stop()

    def _now(self) -> datetime.datetime:
        return datetime.datetime.now(UTC)

    # training

    def train(self, model, train_logger):
        try:
            return self._train(model, train_logger)
        finally:
            with self._lock:
                self._pause_until = time.monotonic() + PAUSE_GRACE_S

    def _train(self, model, train_logger):
        datasets = None
        topic_meta: typing.List[dict] = []
        try:
            topic_meta = _topic_meta()
            datasets = provide_historic_data(datetime.timedelta(days=HISTORY_DAYS))
        except Exception as ex:
            logger.warning(f"history not available: {type(ex).__name__}: {ex}")
        now = clock.now()
        trained = None
        unreadable = datasets is None
        if datasets is not None:
            try:
                trained, notes, history_failed = ray.get(train_anomaly_model.remote(
                    datasets, topic_meta, now, self.config.check_data_anomalies))
                for note in notes:
                    logger.info(note)
                unreadable = history_failed
            except Exception:
                logger.exception("training task failed")
                unreadable = True

        with self._lock:
            if unreadable and self._history_retries < HISTORY_RETRIES:
                # The day stays untrained, so the next training reads the history again and the model stays.
                self._history_retries += 1
                self._retry_at = time.monotonic() + HISTORY_RETRY_S
                logger.warning(f"history unreadable, training again in {HISTORY_RETRY_S // 60} minutes "
                               f"(retry {self._history_retries} of {HISTORY_RETRIES})")
                return None
            self._history_retries = 0
            self._retry_at = None

        if trained is None:
            trained = AnomalyModel(SCHEMA_VERSION, now, {})
        previous = _unwrap(model)
        if unreadable and previous is not None:
            # Out of retries, the day counts as trained; what the last attempt got replaces only those devices.
            trained = AnomalyModel(SCHEMA_VERSION, now, {**previous.devices, **trained.devices})
        # Read by the content test to tell a training on real history from one on nothing.
        if train_logger is not None:
            profiles = list(trained.devices.values())
            try:
                train_logger.log_metrics({
                    "devices_trained": len(profiles),
                    "devices_with_curve_model": sum(isinstance(p.curve, ContCurveModel) for p in profiles),
                    "devices_with_load_model": sum(isinstance(p.curve, LoadCurveModel) for p in profiles),
                })
            except Exception:
                # A metrics store that is down must not keep the trained model from the operator.
                logger.exception("could not log the training metrics")
        return trained

    def need_retraining(self, model) -> bool:
        with self._lock:
            if self._retry_at is not None:
                due = time.monotonic() >= self._retry_at
            else:
                current = _unwrap(model)
                due = current is None or _berlin_date(current.trained_until) != _berlin_date(clock.now())
            if due:
                # Covers the whole blocked time: training, registering, loading and catching up afterwards.
                self._paused = True
                self._pause_until = None
        return due

    # detection

    def infer(self, model, data: typing.Dict[str, typing.Any], selector: str, device_id, timestamp: datetime.datetime):
        results = []
        try:
            results = self._infer(model, data, device_id, timestamp)
        except Exception:
            logger.exception("infer failed")
        return timestamp, (results or None), None

    def _infer(self, model, data, device_id, timestamp) -> typing.List[dict]:
        ts = _aware(timestamp)
        key = str(device_id or data.get("original_input_ids") or "")
        lag = (self._now() - ts).total_seconds()
        with self._lock:
            self._catching_up = lag >= LAG_LIMIT_S
            if self._paused and not self._catching_up:
                self._paused = False
            st = self._live.get(key)
            if st is None:
                st = self._live[key] = live.DeviceState()

        try:
            value = float(data["value"])
        except (KeyError, TypeError, ValueError):
            logger.warning("message without a numeric value")
            return []
        if not math.isfinite(value):
            return []

        anomaly_model = _unwrap(model)
        profile = anomaly_model.profile(key) if anomaly_model is not None else None
        is_load = profile is not None and profile.device_type == "load_device"
        finished = None
        if is_load:
            try:
                finished = st.run.update(ts, value, profile.baseline)[1]
            except Exception:
                logger.exception("run tracking failed")

        results = []
        checks = []
        if self.config.check_data_extreme_outlier:
            checks.append(("extreme value", lambda: extreme_value.check(profile, st, key, value)))
        if self.config.check_receive_time_outlier:
            checks.append(("receive time", lambda: self._under_lock(receive_time.check, profile, st, key, ts)))
        if self.config.check_data_anomalies and profile is not None:
            if isinstance(profile.curve, ContCurveModel):
                checks.append(("curve", lambda: continuous_curve.check(profile.curve, st, key, ts, value)))
            elif isinstance(profile.curve, LoadCurveModel) and finished is not None:
                checks.append(("load curve", lambda: load_curve.check(profile.curve, key, finished, profile.baseline)))
        for name, check in checks:
            try:
                result = check()
            except Exception:
                logger.exception(f"{name} check failed")
                continue
            if result is not None:
                results.append(result)
        return results

    def _under_lock(self, check, *args):
        # The check reads and writes the device state the watchdog shares, so it runs under the lock.
        with self._lock:
            return check(*args)

    # missing messages

    def _watchdog_loop(self):
        while not self._stop_event.wait(WATCHDOG_INTERVAL_S):
            try:
                self._check_outages()
            except Exception:
                logger.exception("outage check failed")

    def _check_outages(self):
        if not self.config.check_receive_time_outlier:
            return
        anomaly_model = _unwrap(getattr(self, "model", None))  # read once; the library swaps the reference
        reports = []
        with self._lock:
            paused = self._paused and (self._pause_until is None or time.monotonic() < self._pause_until)
            if paused or self._catching_up:
                return
            now = self._now()
            for key, st in self._live.items():
                profile = anomaly_model.profile(key) if anomaly_model is not None else None
                report = receive_time.outage(profile, st, key, now)
                if report is not None:
                    reports.append(report)
        for report in reports:  # outside the lock: producing may block
            try:
                self.produce(report)
            except Exception:
                logger.exception("could not report a missing message")

    # schema

    def handle_result_error(self, error, message, produce, device_id):
        # Catches missing middle keys like ENERGY, but not a missing last key like power.
        if not self.config.check_data_schema:
            return
        produce(schema.report(message, device_id))


if __name__ == "__main__":
    from operator_lib.operator_lib import OperatorLib

    op = Operator()
    OperatorLib(op, name="anomaly-detector-operator", git_info_file="git_commit",
                result_error_handler=op.handle_result_error)
