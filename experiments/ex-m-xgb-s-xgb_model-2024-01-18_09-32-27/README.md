
# Experiment: m-xgb-s-xgb_model

## Parameters
- Historical data starting from: 2021-10-01 00:00:00
- Historical data ending at: 2021-10-10 00:00:00
- Streaming data starting from: 2021-10-10 00:00:00
- Streaming data ending at: 2021-11-01 00:00:00
- Time interval: 60
- Concept drift detection strategy: ACTIVE
- Concept drift detection algorithm: DDM
- DDM:

| Attribute | Value |
|---|---|
| Warning Level Factor | 2.0 |
| Drift Level Factor | 3.0 |
| Minimum Numbers of Instances to Start Looking for Changes | 5 |


## Results for running time prediction
### Model Performance Metrics
| Model                               | MAE (s)   | MAPE (%)   | RMSE (s)   |
|-------------------------------------|-----------|------------|------------|
| Base Model (XGBoost)                 | 45.81637290262126 | 40.33081638383994 | 65.506660659676 |
| Base Model with Incremental Learning | 42.9033698807001      | 23.969297488183162      | 65.63694686187527      |

## Results for dwell time prediction
### Model Performance Metrics
| Model                               | MAE (s)   | MAPE (%)   | RMSE (s)   |
|-------------------------------------|-----------|------------|------------|
| Base Model (XGBoost)                 | 11.031494904113444 | 50.88909107177862 | 20.646408296378063 |
| Base Model with Incremental Learning | 4.725115889781399      | 19.60005228830876      | 13.22807286305842      |

    