# Anomaly Detector

Detects anomalies in a numeric series per device: extreme values, unusual gaps between messages and malformed messages.

## Input and output

- Input `value`: a number. Input `original_input_ids` identifies the device when the input comes from another operator; otherwise the device is the message's own device. The time of a value is the time of its message.
- Output, one message per anomaly:

| Name | Meaning |
| --- | --- |
| `type` | `extreme_value`, `time` or `schema` |
| `sub_type` | `high`/`low` for `extreme_value` and `time` |
| `value` | the value, the gap in minutes or the message (`schema`) |
| `mean` | mean of the statistic the value was compared with |
| `threshold` | the bound that was crossed |
| `device_id` | the device |

## Detection

| Check | Config | How | Code |
| --- | --- | --- | --- |
| Extreme value | `check_data_extreme_outlier` | the 0.01 % and 99.99 % quantiles of all values in the history, widened by 20 % of their distance on both sides, at least by 5 % of the larger quantile; for a quantity that never was negative, such as power, the lower bound is 0, so only a negative value counts; reported once per excursion beyond a bound; quantiles because mean ± 5 standard deviations sits inside the upper mode of a two-mode device | [extreme_value.py](algo/detectors/extreme_value.py) |
| Gap between messages | `check_receive_time_outlier` | too long: 3 times the 99.9 % quantile of the gaps in the history, at least 5 minutes, reported by a watchdog while no message comes, once per outage, also across a restart: at start the operator reads the last message of each device back from the history and watches it when its bound had not passed yet; too short: below mean − 5 standard deviations, only for devices that send regularly, reported when the message arrives | [receive_time.py](algo/detectors/receive_time.py) |
| Message schema | `check_data_schema` | a message the input mapping cannot read | [schema.py](algo/detectors/schema.py) |

A device without enough history (30 values or gaps) uses the same statistics over its last 10,000 live values and gaps until the next training, recomputed with every value while it has fewer than 1,000.

## Training

The operator is an `MLOperator` of analytics-operator-lib-python. At start, when no model is registered in MLflow, and once a day (Berlin date) it trains on 50 days of the history of its inputs. The model holds one profile per device and is stored in MLflow. The training runs as a Ray task on the cluster (`algo/training.py`); the image starts the operator with `uv run`, so Ray builds the environment of `uv.lock` on the workers. There is no learning phase. The last arrivals read back at start are read in the pod over the database connection, not in a Ray task, so a busy cluster does not delay the start of consuming; the history is read in a Ray task only for an input whose history is in Kafka alone, or after a read in the pod failed for another reason than its time limit.

A history that cannot be read is tried again every 10 minutes, six times; after that the day counts as trained and the previous profiles stay. An input from another operator has only the retention of its Kafka topic as history.

## Code

[`model.py`](algo/model.py) holds the shared types of a model and the profile of a device, [`history.py`](algo/history.py) reads and groups the history, [`training.py`](algo/training.py) builds the profiles from it, [`live.py`](algo/live.py) keeps the runtime state of a device, and [`main.py`](main.py) routes each message to the detectors and runs the watchdog.

## Configuration

| Name | Default | Meaning |
| --- | --- | --- |
| `check_data_extreme_outlier` | true | report extreme values |
| `check_receive_time_outlier` | true | report unusual gaps between messages |
| `check_data_schema` | true | report malformed messages |
| `logger_level` | warning | `debug`, `info`, `warning`, `error`, `critical` |

An empty value means the default. The configuration values of version 1 (`init_phase_*`, `train_*`, `retrain`, `curve_detector_training_mode`) and `check_data_anomalies` of version 2 are ignored.

## Versions

Every build on `main` pushes `:vX.Y.Z`; a branch `release/vN` builds only `:vN.x.y`. See `.github/workflows/prod.yml`. `:prod` stays on the last build of version 1, which older pipelines run; `.github/workflows/retag.yml` points a tag at an existing version or digest.
