# Anomaly Detector

Reports three kinds of anomaly in a numeric series per device: a value outside the usual range of the device, a device that stops sending, and a message the input mapping cannot read.

The check of extreme values suits momentary readings such as power, temperature or flow. The reading of a meter only grows, and the daily training moves its bounds along: a meter is reported when it is reset or jumps far ahead, but not for a high consumption. The check of missing messages suits both. To watch the consumption of a meter, give the operator the difference between consecutive readings.

## Input and output

- Input `value`: a number. Input `original_input_ids` identifies the device when the input comes from another operator; otherwise the device is the message's own device. The time of a value is the time of its message.
- Output, one message per anomaly:

| Name | Meaning |
| --- | --- |
| `type` | `extreme_value`, `time` or `schema` |
| `sub_type` | `high` or `low` for `extreme_value`, `high` for `time`, empty for `schema` |
| `value` | the value, the minutes without a message, or the message as JSON (`schema`) |
| `mean` | mean of the values, or of the gaps in minutes |
| `threshold` | the bound that was crossed |
| `device_id` | the device |

## Detection

| Check | Config | Rule |
| --- | --- | --- |
| Extreme value | `check_data_extreme_outlier` | Beyond the range of the device's values without their 5 most extreme values on each side (at most one value in 200), widened by half that range on both sides and at least by 5 % of its larger end. For a quantity that never was negative, such as power, the lower bound stops at 0: a device whose values reach 0 is low only for a negative value, a device with a base load also when it drops below that load by more than the widening. Reported once per excursion beyond a bound; an excursion ends with the first value within the bounds an hour or more after the last value beyond them, so a value swinging around a bound is reported once. A number of values rather than a quantile, so that the bounds mean the same for a device that sends every minute and one that sends every hour, and an outlier in the history does not hide its own repetition; the range rather than mean ± 5 standard deviations, which lies inside the upper mode of a device with two modes, such as standby and a heating element. |
| Missing messages | `check_receive_time_outlier` | No message for longer than 3 times the 99.9 % quantile of the device's gaps, at least 5 minutes. A watchdog checks every 5 seconds and reports once per outage; the next message watches the device again. The silence counts from the last message the operator received of the device, so working off a backlog is no outage, and a late message of one device hides no other. |
| Malformed message | `check_data_schema` | A message the input mapping cannot read; reported once per device until it sends a readable message. |

The statistics are in [`algo/stats.py`](algo/stats.py), the checks in [`main.py`](main.py). A device without a profile, or with fewer than 30 values in its history, is judged by its last 10,000 live values and gaps; they are recomputed every 100 values, and with every value while there are fewer than 1,000.

## Training

The operator is an `MLOperator` of analytics-operator-lib-python. At start, when no model is registered in MLflow, and once a day (Berlin date) it learns from 50 days of the history of its inputs. The model holds the statistics of every device and is stored in MLflow; the training also logs the bounds of every device as the table `profiles.json` and their number as the metric `devices_trained`.

The history is read over the database connection in the pod, at most 5 minutes, and the statistics are computed there: they take milliseconds per device. A Ray task reads and computes it only for an input whose history is in Kafka alone, or after a read in the pod failed for another reason than its time limit; the image starts the operator with `uv run`, so Ray builds the environment of `uv.lock` on the workers. A history that cannot be read is tried again every 10 minutes, six times; after that the day counts as trained and the previous profiles stay.

Consuming waits while the operator trains. The watchdog pauses from the start of a training until the first message afterwards that is less than 2 minutes old, at most 10 minutes after the training. At start, the last message of every device is read back from up to 2 days of history, in the pod within 2 minutes and with the same Ray task as fallback, so an outage that began before a restart is still reported; a device whose bound had passed by the start is left to the operator before.

## Code

[`main.py`](main.py) handles the messages, runs the watchdog and the training; [`algo/stats.py`](algo/stats.py) holds the statistics and their bounds; [`algo/history.py`](algo/history.py) reads and groups the history; [`algo/training.py`](algo/training.py) derives the statistics of each device from it; [`algo/model.py`](algo/model.py) is the model stored in MLflow.

[`scripts/replay_history.py`](scripts/replay_history.py) replays the history of real devices through the operator and counts its reports, also with injected anomalies. It measures; it is not a test.

[`docs/known-limitations-and-open-decisions.md`](docs/known-limitations-and-open-decisions.md) lists what is not verified yet and what is still to be decided.

## Configuration

| Name | Default | Meaning |
| --- | --- | --- |
| `check_data_extreme_outlier` | true | report extreme values |
| `check_receive_time_outlier` | true | report devices that stop sending |
| `check_data_schema` | true | report malformed messages |
| `logger_level` | warning | `debug`, `info`, `warning`, `error`, `critical` |

An empty value means the default. The configuration values of version 1 (`init_phase_*`, `train_*`, `retrain`, `curve_detector_training_mode`) and `check_data_anomalies` of version 2 are ignored.

## Development

```sh
uv sync
uv run python -m unittest discover -s tests -t .
uv run --env-file .env main.py  # locally, with CONFIG and the Kafka settings in .env
```

## Versions

Every build on `main` pushes `:vX.Y.Z`; a branch `release/vN` builds only `:vN.x.y`. See `.github/workflows/prod.yml`. `:prod` stays on the last build of version 1, which older pipelines run; `.github/workflows/retag.yml` points a tag at an existing version or digest.
