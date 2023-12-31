
# Experiment: m-xgb-s-xgb_model

## Parameters
- Historical data starting from: 2021-10-01 00:00:00
- Historical data ending at: 2021-10-10 00:00:00
- Streaming data starting from: 2021-10-10 00:00:00
- Streaming data ending at: 2022-11-01 00:00:00
- Time interval: 60
- Concept drift detection strategy: ACTIVE
- Concept drift detection algorithm: DDM
- DDM:

| Attribute | Value |
|---|---|
| Warning Level Factor | 2.0 |
| Drift Level Factor | 3.0 |
| Minimum Numbers of Instances to Start Looking for Changes | 5 |


## Results
### Model Performance Metrics
| Model                               | MAE (s)   | MAPE (%)   | RMSE (s)   |
|-------------------------------------|-----------|------------|------------|
| Base Model (XGBoost)                 | 10.973392029746565 | 58.004968550866124 | 18.700606886616313 |
| Base Model with Incremental Learning | 1.9168827846573362      | 7.7964432257389875      | 6.748026876510337      |

    