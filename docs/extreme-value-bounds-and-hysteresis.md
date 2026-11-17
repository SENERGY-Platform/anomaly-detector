# Bounds and hysteresis of extreme values

## Scope

How the check of extreme values sets its bounds and when an excursion beyond them ends, as of version 3.1, and the measurement on public data the choice rests on: `VALUE_IGNORED`, `VALUE_IGNORED_PER`, `VALUE_MARGIN` and `VALUE_MIN_MARGIN` in [`algo/stats.py`](../algo/stats.py), `EXCURSION_END` and `Operator._extreme` in [`main.py`](../main.py). Re-check it when one of them changes.

The measurement covers momentary readings and consumption per hour, sent every minute, every 15 minutes or every hour. It does not cover meter readings, for which the check is not made (see the [README](../README.md)), devices that send less often than every hour, the devices of the platform, nor the check of missing messages: the public data have no real gaps. Whether the numbers hold for a kind of device not listed below cannot be assumed from here.

## Decision

- **Bounds:** the range of a device's values in its history without their 5 most extreme values on each side, but at most one value in 200, widened by half that range on both sides and at least by 5 % of its larger end. With fewer than 200 values, as in the live statistic of a new device at first, no value is ignored.
- **End of an excursion:** the first value within the bounds an hour or more after the last value beyond them, by message time. Until then a value beyond the same bound belongs to the same excursion and is not reported again.

Version 3.0 used the 0.01 % and 99.99 % quantiles, widened by a fifth of their distance, and ended an excursion with the first value within the bounds.

## Why

A fixed quantile means a different value at a different rate. 50 days of a device hold about 1,200 values when it sends every hour, 4,800 every 15 minutes and 72,000 every minute. The 99.99 % quantile is practically the maximum of the first two and about the 8th largest value of the third:

- Every hour or every 15 minutes, a single outlier in the history sets the bound, and the same outlier is not reported again until it has left the history 50 days later.
- Every minute, the bounds lie so close that the loudest device reported every 2.4 days.

A number of values means the same at every rate. An excursion that ends with the first value back within the bounds reports a value swinging around a bound at every swing, such as a heat pump whose heating element runs near the upper bound.

## Measurement

### Data

| Set | Content | Rate | Series | License |
| --- | --- | --- | --- | --- |
| [LEAD1.0](https://github.com/samy101/lead-dataset), small | electricity per hour of commercial buildings, 2016, with anomalies labelled by hand | 1 h | 200 | CC BY 4.0 according to its [paper](https://arxiv.org/abs/2203.17256); the repository has no license file |
| [WPuQ](https://zenodo.org/records/5642902), 2019, 15 min | power of 37 heat pumps, 37 households, 3 PV inverters and a transformer in one district | 15 min | 78 | CC BY 4.0 |
| [REFIT](https://zenodo.org/records/5063428), cleaned, houses 2 and 11 | power of the house and of 9 appliances each, every 8 s | 1 min and 15 min | 40 | CC BY 4.0 |

REFIT was reduced to one value per minute and per 15 minutes by taking the first value of each interval, like a device that reports its momentary reading. A mean would smooth exactly the outliers away.

### Method

Like the operator: every day the bounds come from the 50 days before, and the values of the day run through the logic of `Operator._extreme`. Per series 315 to 518 days were judged. The logic was rebuilt vectorised to compare variants. Against the operator of this repository it gave identical reports on 285 windows of 10 or 30 days, 95 reports in all, among them three windows in which the hour suppressed 46 swings.

- **Reports:** per device and day. Only LEAD has labels, so elsewhere a report is not necessarily a false alarm.
- **Swing:** a report whose excursion follows the previous one beyond the same bound after at most 3 values within.
- **Injected outlier:** a single value of 1.5, 2 or 3 times the maximum of the 50 days before, every 14 days, and the same value again 7 days later, when the first one is part of the history.

### Results

Bounds, with the end of an excursion of version 3.0. The table gives the days between two reports of the loudest device, more is quieter, and the share of repeated 1.5-fold outliers that were reported:

| Set | 99.99 % + 20 % (3.0) | 5 values + 50 % (3.1) | 3 values + 50 % |
| --- | --- | --- | --- |
| LEAD, 1 h | 44 days, **0 %** | 18 days, 100 % | 18 days, 100 % |
| Heat pumps, 15 min | 17 days, **0 %** | 12 days, 100 % | 14 days, 100 % |
| Households, 15 min | 26 days, **1 %** | 19 days, 100 % | 29 days, 99 % |
| REFIT, 15 min | 23 days, **1 %** | 11 days, 100 % | 15 days, 99 % |
| REFIT, 1 min | **2.4 days**, 99 % | 13 days, 99 % | 17 days, 98 % |

With version 3.0, LEAD did not report a repeated outlier of 3 times the maximum either. With 5 values + 50 %, none of the 318 series reported more than once a week, against one with version 3.0. Every first outlier of 1.5 times the maximum was reported, with all three variants.

End of an excursion, with 5 values + 50 %. The table gives the reports of all series and the share of swings among them:

| End | Reports | Swings, heat pumps | Swings, households | Swings, REFIT 1 min |
| --- | --- | --- | --- | --- |
| first value within (3.0) | 1,084 | 18 % | 13 % | 6 % |
| first value within the bounds without widening | 1,051 | 18 % | 3 % | 5 % |
| **first value within an hour after the last beyond (3.1)** | 997 | 0 % | 1 % | 0 % |
| no report of the same side for 6 hours after a report | 941 | 2 % | 1 % | 0 % |

No end variant lost an injected outlier.

Two thirds of the reports on LEAD were low ones: drops below the base load of a building.

### Rejected

- **Other fixed quantiles** (99.9 %, 99.5 %, 99 % with 20 %, 50 % or 100 % widening). They keep the dependence on the rate. 99.5 % + 50 % reported more than once a week for 27 % of the heat pumps.
- **3 instead of 5 values.** Quieter, and as good against a single repeated outlier. But a history with four or five outliers, such as a heating element that runs a few times in 50 days, would set the bound again.
- **End within the bounds without widening.** No effect on a device with two modes, whose upper end lies close to the upper bound.
- **No report of the same side for 6 hours.** About as quiet, but it suppresses a new event within the 6 hours, and the hour does not.

### Limits

- **Few labels.** Except on LEAD, the numbers count reports, not false alarms; some reports may be real anomalies, such as a heating element that rarely runs. Of the 2,972 anomalies labelled on LEAD, every variant reported about a tenth: most are changes of shape or level within the usual range, not extreme values.
- **Injected outliers are single values.** A longer anomaly was not injected.
- **Cleaned data.** The cleaned REFIT data have values above 4,000 W set to 0 and short gaps filled.
- **Hourly devices still swing.** For a device that sends every hour, the hour is one message, so a swing there is still reported. That was 4 % of the reports on LEAD.
- **Training day.** The measurement retrained at midnight UTC, the operator retrains by the date in Berlin.

The scripts of the measurement are not kept; the method above is enough to repeat it.
