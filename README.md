# Anomaly Detector


## Input
| key                | type                                                 | description                                               | 
|--------------------|------------------------------------------------------|-----------------------------------------------------------|----------|
|     |                                              |                      |     |
| `value`   | string | Anything that outputs a numeric value |

## Output 

| key                | type                                                 | description                                               | 
|--------------------|------------------------------------------------------|-----------------------------------------------------------|----------|
|     |                                              |                      |     |
| `value`   | string | ID of source providing weather forecast data. |
| `type`    | string | Anomaly Type |
| `sub_type` | string | Anomaly Sub Type |
| `mean` | float | Current mean of point outlier detector that detected the anomaly |
| `threshold` | float | Threshold that was used to compare with single values |
| `device_id` | string | Device for which the anomaly was detected |
| `initial_phase` | string | Message whether and how long the operator is in the initialization/training phase |

## Config options
| key                | type                                                 | description                                               | required |
|--------------------|------------------------------------------------------|-----------------------------------------------------------|----------|
|     |                                              |                      |     |
| `logger_level`     | string                                               | `info`, `warning` (default), `error`, `critical`, `debug` | no       |
| `data_path`        | string                                               | Path to reward and model files. Default: "/opt/data"      | no       |

