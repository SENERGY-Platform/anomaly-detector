# Known limitations and open decisions

## Scope

What the operator does not do yet, or does in a way that is still to be decided, as of version 3: the version that dropped the curve detector and learns its statistics in the pod with `provide_historic_frames_local` of analytics-operator-lib-python v1.14.0, with a Ray task as fallback. Each point names the code it is about; re-check a point when that code changes, and the points about the library when its version changes. The curve detector of versions 1 and 2 is gone and not covered here.

## Not verified yet

- **Real device data.** The bounds and the reporting rules have only been tested on synthetic data. [`scripts/replay_history.py`](../scripts/replay_history.py) replays the history of real devices and counts the reports per day, also with injected anomalies. The open decisions on extreme values below wait for that measurement.
- **The training in a real deployment.** The replay does not go through a deployment: reading 50 days in the pod with the deployment's own configuration, matching the device keys of the history with those of the messages, and the time the read takes. The time limit of 5 minutes (`HISTORY_TIMEOUT_S` in [`main.py`](../main.py)) is not measured. A run in a test pipeline with `logger_level` set to `info` logs a line per device with the number of values read; the training run in MLflow holds the bounds of every device in the table `profiles.json`.

## Open decisions

### Hysteresis of extreme values

An excursion ends as soon as one value lies within the bounds again (`Operator._extreme` in [`main.py`](../main.py)). A value that oscillates around a bound is reported at every new crossing. A hysteresis would end the excursion only some way inside the bound, or after some time there. Decide once the measurement on real devices shows how often this happens.

### Bounds of extreme values

The bounds are the 0.01 % and 99.99 % quantiles of 50 days of values, widened by 20 % of their distance ([`algo/stats.py`](../algo/stats.py)). For a device that sends every 15 minutes, about 4,800 values, these quantiles are practically the minimum and the maximum: a single outlier in the history sets the bound for 50 days, and the same anomaly is not reported again until it has left the history. Less extreme quantiles with a wider margin would be robust against that, but the margin needs the measurement.

### A device whose key the history does not know

The device key of a message (`device_id`, else `original_input_ids`) and the key of its history (the device column, the `original_input_ids` column or the filter value of the input, [`algo/history.py`](../algo/history.py)) are derived separately. When they do not match, the device is silently judged by its live values, as if it had no history. A warning, once per device, when a device without a profile sends although the model holds profiles would make such a mismatch visible.

### Type and precision of the output

`value` is declared a string in the manifest, but carries a number for `extreme_value` and `time`, and the message as JSON for `schema`. `mean` and `threshold` are rounded to two decimals, too coarse for a quantity in kWh or MWh. Changing either breaks consumers of the output and needs a new major version.

### Endpoint in the replay script

[`scripts/replay_history.py`](../scripts/replay_history.py) has the endpoint of the timescale-wrapper as a constant. It should be a required argument or an environment variable, so that the repository names no deployment.

## Known limitations

- **A long backlog after a restart.** The silence of a device counts from the last message the operator received of it, and at start from its last message in the history. When the backlog after a restart takes longer to work off than the bound of a device whose messages lie deep in it, that device is reported although it sent. The pause of the watchdog covers only a training, not the start: a pause at the start would delay the report of an outage that began before it.
- **Consuming waits for a training.** The library runs the training in the path that consumes, once a day and after a failed read every 10 minutes, at most six times. The watchdog pauses meanwhile; how long consuming waits depends on the read, see above.

## Limits of the operator library

All are in [analytics-operator-lib-python](https://github.com/SENERGY-Platform/analytics-operator-lib-python) v1.14.0, not in this repository.

- `provide_historic_frames_local` raises `HistoryNeedsRayError` for the whole call as soon as one input has its history only in Kafka, such as the output of another operator. The training then reads all inputs in a Ray task and waits for a free worker again. Skipping such inputs one by one would keep the others in the pod.
- `MLOperator` connects to the Ray cluster before every training, also when the training does not use it. The cluster has to be reachable, the image has to run the Python of the cluster (3.10, past its end of life in October 2026), and it carries Ray and pyarrow, about 360 MB of about 1 GB. An `MLOperator` that connects only when a training asks for Ray would remove all three.
