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

import concurrent.futures
import datetime
import json
import math
import threading
import time
import typing
import zoneinfo

import ray

from operator_lib import util
from operator_lib.util import MLOperator, Config, Inputs, Outputs, logger
from operator_lib.util import clock
from operator_lib.util.helpers import provide_historic_data, provide_historic_frames_local

from algo.history import last_arrivals, read_last_arrivals
from algo.model import SCHEMA_VERSION, AnomalyModel, DeviceProfile
from algo.stats import SampleBuffer, gap_stats, value_stats
from algo.training import fit_anomaly_model, profile_table, train_anomaly_model

__all__ = ("Operator", "CustomConfig", "OperatorInputs", "OperatorOutputs")

HISTORY_DAYS = 50
# The training reads the history in the pod at most this long; consuming waits meanwhile.
HISTORY_TIMEOUT_S = 300
# A requested history that cannot be read, as when the Ray cluster has no CPU free, is tried again this
# often, this far apart, before the day counts as trained on what could be read.
HISTORY_RETRY_S = 600
HISTORY_RETRIES = 6
# After a training, the first message less old than this (wall clock minus Kafka time) shows that the backlog
# of the training is worked off.
LAG_LIMIT_S = 120
WATCHDOG_INTERVAL_S = 5
# After a training the watchdog stays silent only this long for a message that does not come.
PAUSE_GRACE_S = 600
# At start the last arrivals are read back from this much history beyond the widest outage bound, at most
# RESTORE_MAX: a pod that replaces another during an outage has not seen the message before it.
RESTORE_MARGIN = datetime.timedelta(minutes=10)
RESTORE_MAX = datetime.timedelta(days=2)
RESTORE_TIMEOUT_S = 120
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

    check_data_extreme_outlier: bool = True
    check_data_schema: bool = True
    check_receive_time_outlier: bool = True

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        for name in ("check_data_extreme_outlier", "check_data_schema", "check_receive_time_outlier"):
            setattr(self, name, _parse_bool(getattr(self, name), getattr(type(self), name)))


class OperatorInputs(Inputs):
    value: float
    original_input_ids: str


class OperatorOutputs(Outputs):
    type: str
    sub_type: str
    value: str
    mean: float
    threshold: float
    device_id: str


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


def _frames_in_pod(duration: datetime.timedelta, timeout_s: float):
    """The history of the inputs as pandas frames, read in the pod, at most timeout_s long: the library sets no
    timeout on the database connection, and consuming waits meanwhile. A read that runs out goes on in its thread,
    and its result is dropped."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="history")
    try:
        return pool.submit(provide_historic_frames_local, duration).result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False)


def _aware(t: datetime.datetime) -> datetime.datetime:
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def _berlin_date(t: datetime.datetime) -> datetime.date:
    return _aware(t).astimezone(BERLIN).date()


def _minutes(delta: datetime.timedelta) -> float:
    return delta.total_seconds() / 60


class Device:
    """What the operator keeps of one device; shared with the watchdog, so only touched under the operator's lock."""

    def __init__(self):
        # The last values and gaps, which judge a device without a profile.
        self.values = SampleBuffer(value_stats)
        self.gaps = SampleBuffer(gap_stats)
        self.last_arrival: typing.Optional[datetime.datetime] = None  # message time of its newest message
        self.seen: typing.Optional[datetime.datetime] = None  # when the operator last received a message of it
        self.armed = False  # whether its silence may be reported, once per outage
        self.extreme_side: typing.Optional[str] = None  # "high" or "low" while its values are beyond a bound


class Operator(MLOperator):
    configType = CustomConfig
    inputs = OperatorInputs
    outputs = OperatorOutputs

    def __init__(self):
        # Guards the devices the watchdog thread shares with infer(), and the flags below.
        self._lock = threading.Lock()
        self._devices: typing.Dict[str, Device] = {}
        # The devices whose malformed message was reported and which sent no readable one since.
        self._malformed: typing.Set[str] = set()
        self._paused = False  # training or its aftermath: no missing-message reports
        self._pause_until: typing.Optional[float] = None  # monotonic end of the pause once the training is over
        self._retry_at: typing.Optional[float] = None  # monotonic time of the next training after an unreadable history
        self._history_retries = 0
        self._stop_event = threading.Event()
        self._watchdog: typing.Optional[threading.Thread] = None

    # lifecycle

    def init(self, *args, **kwargs):
        started = self._now()
        super().init(*args, **kwargs)
        # Here, before the library starts consuming, nothing else can start or stop Ray meanwhile.
        if self.config.check_receive_time_outlier and not self.config.test_end:
            self._restore_arrivals(started)
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
        now = clock.now()
        trained, unreadable = self._fit(now)

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
            try:
                train_logger.log_metrics({"devices_trained": len(trained.devices)})
                train_logger.log_table(profile_table(trained), "profiles.json")
            except Exception:
                # A metrics store that is down must not keep the trained model from the operator.
                logger.exception("could not log the training metrics")
        return trained

    def _fit(self, now: datetime.datetime) -> typing.Tuple[typing.Optional[AnomalyModel], bool]:
        """
        The model from the history and whether a requested history could not be read. Read and computed in the
        pod: the statistics take milliseconds per device. In a Ray task only for an input whose history only Kafka
        has, or after a read in the pod failed for another reason than its time limit.
        """
        duration = datetime.timedelta(days=HISTORY_DAYS)
        try:
            topic_meta = _topic_meta()
        except Exception as ex:
            logger.warning(f"history not available: {type(ex).__name__}: {ex}")
            return None, True
        try:
            frames = _frames_in_pod(duration, HISTORY_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            logger.warning(f"history not read in the pod within {HISTORY_TIMEOUT_S} s")
            return None, True
        except Exception as ex:
            # An input only Kafka has (HistoryNeedsRayError), or one input that cannot be read, which fails the
            # whole read here; the Ray task reads the others and leaves only that one out.
            logger.info(f"history not read in the pod ({type(ex).__name__}: {ex}), reading it in a Ray task")
            return self._fit_in_ray(duration, topic_meta, now)
        trained, notes, failed = fit_anomaly_model(frames, topic_meta, now)
        for note in notes:
            logger.info(note)
        return trained, failed

    @staticmethod
    def _fit_in_ray(duration, topic_meta, now) -> typing.Tuple[typing.Optional[AnomalyModel], bool]:
        # MLOperator has connected Ray for train() already.
        try:
            trained, notes, failed = ray.get(train_anomaly_model.remote(provide_historic_data(duration), topic_meta,
                                                                        now))
        except Exception:
            logger.exception("training task failed")
            return None, True
        for note in notes:
            logger.info(note)
        return trained, failed

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
            self._malformed.discard(str(device_id or ""))
            if self._paused and lag < LAG_LIMIT_S:
                self._paused = False
            device = self._devices.get(key)
            if device is None:
                device = self._devices[key] = Device()

        try:
            value = float(data["value"])
        except (KeyError, TypeError, ValueError):
            logger.debug(f"message of {key} without a numeric value")
            return []
        if not math.isfinite(value):
            return []

        anomaly_model = _unwrap(model)
        profile = anomaly_model.profile(key) if anomaly_model is not None else None

        with self._lock:
            if self.config.check_receive_time_outlier:
                self._arrive(device, profile, ts, self._now())
            if self.config.check_data_extreme_outlier:
                result = self._extreme(key, device, profile, value)
                if result is not None:
                    return [result]
        return []

    @staticmethod
    def _arrive(device: Device, profile: typing.Optional[DeviceProfile], at: datetime.datetime,
                now: datetime.datetime):
        # Also for a message out of order: while the operator works off a backlog, the device is not silent.
        device.seen = now
        last = device.last_arrival
        if last is not None and at <= last:
            return  # out of order or a duplicate: neither a gap nor a newer message
        if last is not None and (profile is None or profile.gaps is None):
            device.gaps.add(_minutes(at - last))
        device.last_arrival = at
        device.armed = True

    @staticmethod
    def _extreme(key: str, device: Device, profile: typing.Optional[DeviceProfile], value: float):
        if profile is not None and profile.values is not None:
            stats = profile.values
        else:
            stats = device.values.stats()
            device.values.add(value)
        if stats is None:
            return None
        side = "high" if value > stats.upper() else "low" if value < stats.lower() else None
        previous, device.extreme_side = device.extreme_side, side
        if side is None or side == previous:
            return None  # once per excursion: a device stuck beyond a bound would report every message
        threshold = stats.upper() if side == "high" else stats.lower()
        return {"type": "extreme_value", "sub_type": side, "value": value, "mean": round(stats.mean, 2),
                "threshold": round(threshold, 2), "device_id": key}

    # missing messages

    def _watchdog_loop(self):
        while not self._stop_event.wait(WATCHDOG_INTERVAL_S):
            try:
                self._check_outages()
            except Exception:
                logger.exception("outage check failed")

    def _restore_arrivals(self, started: datetime.datetime):
        """
        Arms every device whose last message in the history lies within its outage bound of the start, so an
        outage that began before a restart is still reported. One whose bound had passed by the start is left
        alone: the pod before may have reported it already. A device with a live message keeps that one.
        """
        try:
            anomaly_model = _unwrap(getattr(self, "model", None))
            if anomaly_model is None:
                return
            bounds = {key: datetime.timedelta(minutes=p.gaps.upper())
                      for key, p in anomaly_model.devices.items() if p.gaps is not None}
            if not bounds:
                return
            duration = min(max(bounds.values()) + RESTORE_MARGIN, RESTORE_MAX)
            try:
                # Over the database connection, in the pod: a Ray task waits for a free worker of the cluster,
                # minutes on a busy one, while the consumer has not started yet.
                frames = _frames_in_pod(duration, RESTORE_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                raise
            except Exception as ex:
                # An input only Kafka has (HistoryNeedsRayError), or one input that cannot be read, which fails
                # the whole read here; the Ray task reads the others and leaves only that one out.
                logger.info(f"history not read in the pod ({type(ex).__name__}: {ex}), reading it in a Ray task")
                arrivals, notes, failed = self._last_arrivals_in_ray(duration)
            else:
                arrivals, notes, failed = last_arrivals(frames, _topic_meta())
            for note in notes:
                logger.info(note)
            if failed:
                logger.warning("the history of a topic is not readable, its devices keep no last arrival")
            restored = 0
            with self._lock:
                for key, arrival in arrivals.items():
                    if key not in bounds:
                        continue
                    last = _aware(arrival.to_pydatetime())
                    if last + bounds[key] <= started:
                        continue
                    device = self._devices.get(key)
                    if device is None:
                        device = self._devices[key] = Device()
                    if device.last_arrival is None:
                        device.last_arrival = device.seen = last
                        device.armed = True
                        restored += 1
            logger.info(f"restored the last arrival of {restored} devices")
        except Exception:
            logger.exception("could not restore the last arrivals")

    def _last_arrivals_in_ray(self, duration: datetime.timedelta):
        """The last arrivals read in a Ray task, for an input whose history only Kafka has."""
        ray.init(address=self.config.ray_url)
        try:
            task = read_last_arrivals.remote(provide_historic_data(duration), _topic_meta())
            try:
                return ray.get(task, timeout=RESTORE_TIMEOUT_S)
            except Exception:
                ray.cancel(task)
                raise
        finally:
            ray.shutdown()

    def _check_outages(self):
        if not self.config.check_receive_time_outlier:
            return
        anomaly_model = _unwrap(getattr(self, "model", None))  # read once; the library swaps the reference
        reports = []
        with self._lock:
            paused = self._paused and (self._pause_until is None or time.monotonic() < self._pause_until)
            if paused:
                return
            now = self._now()
            for key, device in self._devices.items():
                if not device.armed or device.seen is None:
                    continue
                profile = anomaly_model.profile(key) if anomaly_model is not None else None
                stats = profile.gaps if profile is not None and profile.gaps is not None else device.gaps.stats()
                # Counted from the last message received, not from its message time: a device whose messages
                # come late is not silent, and a late message of one device hides no other.
                silent = _minutes(now - device.seen)
                if stats is not None and silent > stats.upper():
                    device.armed = False  # armed again by its next message
                    reports.append({"type": "time", "sub_type": "high", "value": round(silent, 2),
                                    "mean": round(stats.mean_min, 2), "threshold": round(stats.upper(), 2),
                                    "device_id": key})
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
        with self._lock:
            # Once until the device sends a readable message: a misconfigured device sends nothing else.
            if str(device_id or "") in self._malformed:
                return
            self._malformed.add(str(device_id or ""))
        produce({"type": "schema", "sub_type": "", "value": json.dumps(message), "device_id": device_id})


if __name__ == "__main__":
    from operator_lib.operator_lib import OperatorLib

    op = Operator()
    OperatorLib(op, name="anomaly-detector-operator", git_info_file="git_commit",
                result_error_handler=op.handle_result_error)
