# Anomaly Detector

Detects anomalies in a numeric series per device: extreme values, unusual gaps between messages, malformed messages and, optionally, unusual curves.

## Input and output

- Input `value`: a number. Input `original_input_ids` identifies the device when the input comes from another operator; otherwise the device is the message's own device. The time of a value is the time of its message.
- Output, one message per anomaly:

| Name | Meaning |
| --- | --- |
| `type` | `extreme_value`, `time`, `schema` or `curve_anomaly` |
| `sub_type` | `high`/`low` for `extreme_value` and `time`; `continous_device`, `uncontinious_device_curve` or `uncontinious_device_length` for `curve_anomaly` |
| `value` | the value, the gap in minutes, the message (`schema`) or the start of the curve window |
| `mean` | mean of the statistic the value was compared with |
| `threshold` | the bound that was crossed |
| `device_id` | the device |
| `start_time`, `end_time` | the curve window or the run, ISO 8601 UTC |
| `original_reconstructed_curves` | JSON `[[ms, smoothed, reconstructed], ...]` of the curve window |
| `initial_phase` | declared for the existing export, no longer set |

## Detection

| Check | Config | How | Code |
| --- | --- | --- | --- |
| Extreme value | `check_data_extreme_outlier` | the 0.01 % and 99.99 % quantiles of all values in the history, widened by 20 % of their distance on both sides, at least by 5 % of the larger quantile; for a quantity that never was negative, such as power, the lower bound is 0, so only a negative value counts; reported once per excursion beyond a bound; quantiles because mean ± 5 standard deviations sits inside the upper mode of a two-mode device | [extreme_value.py](algo/detectors/extreme_value.py) |
| Gap between messages | `check_receive_time_outlier` | too long: 3 times the 99.9 % quantile of the gaps in the history, at least 5 minutes, reported by a watchdog while no message comes, once per outage, also across a restart: at start the operator reads the last message of each device back from the history and watches it when its bound had not passed yet; too short: below mean − 5 standard deviations, only for devices that send regularly, reported when the message arrives | [receive_time.py](algo/detectors/receive_time.py) |
| Message schema | `check_data_schema` | a message the input mapping cannot read | [schema.py](algo/detectors/schema.py) |
| Curve, continuous device | `check_data_anomalies` | an autoencoder reconstructs the smoothed last 205 minutes once a minute; an error above the 99.9 % quantile of the errors on the newest fifth of the history, which the network was not trained on, is a hit, and the second hit within 30 minutes is reported; after a report the device is silent for 205 minutes | [continuous_curve.py](algo/detectors/continuous_curve.py) |
| Curve, load device | `check_data_anomalies` | when a run ends, its length and its curve are compared with the median of the runs in the history; a deviation beyond 5 times the scaled MAD, at least 5 minutes in length and 0.05 (normalised) in curve, is reported; runs are only judged when at least 80 % of the runs in the history are within 25 % (at least 5 minutes) of the median length, so a device without a typical run, such as a TV, gets no run check | [load_curve.py](algo/detectors/load_curve.py) |

On producers such as PV the curve check reports cloudy days as an unusual curve; whoever does not want that turns `check_data_anomalies` off.

A device is a load device when, on the minute grid of its history, the 5 % quantile is not negative and at most 10 % of the 95 % quantile (the maximum if the 95 % quantile is not above the 5 % one), and at least half of the values lie within a tenth of the way between the two; otherwise it is a continuous device. A device without enough history (30 values or gaps) uses the same statistics over its last 10,000 live values and gaps until the next training, recomputed with every value while it has fewer than 1,000.

## Training

The operator is an `MLOperator` of analytics-operator-lib-python. At start, when no model is registered in MLflow, and once a day (Berlin date) it trains on 50 days of the history of its inputs. The model holds one profile per device and is stored in MLflow. The training runs as a Ray task on the cluster (`algo/training.py`); the image starts the operator with `uv run`, so Ray builds the environment of `uv.lock`, torch included, on the workers. There is no learning phase.

A history that cannot be read is tried again every 10 minutes, six times; after that the day counts as trained and the previous profiles stay. An input from another operator has only the retention of its Kafka topic as history, so its curve model rarely reaches the three days it needs.

## Code

[`model.py`](algo/model.py) holds the shared types of a model and the profile of a device, [`history.py`](algo/history.py) reads and groups the history, [`training.py`](algo/training.py) builds the profiles from it, [`live.py`](algo/live.py) keeps the runtime state of a device, [`preprocessing.py`](algo/preprocessing.py) provides the minute grid, smoothing, windows and load runs, and [`main.py`](main.py) routes each message to the detectors and runs the watchdog.

## Configuration

| Name | Default | Meaning |
| --- | --- | --- |
| `check_data_extreme_outlier` | true | report extreme values |
| `check_receive_time_outlier` | true | report unusual gaps between messages |
| `check_data_schema` | true | report malformed messages |
| `check_data_anomalies` | false | train curve models and report curve anomalies |
| `logger_level` | warning | `debug`, `info`, `warning`, `error`, `critical` |

An empty value means the default. The configuration values of version 1 (`init_phase_*`, `train_*`, `retrain`, `curve_detector_training_mode`) are ignored.

## Versions

Every build on `main` pushes `:vX.Y.Z`; a branch `release/vN` builds only `:vN.x.y`. See `.github/workflows/prod.yml`. `:prod` stays on the last build of version 1, which older pipelines run; `.github/workflows/retag.yml` points a tag at an existing version or digest.
