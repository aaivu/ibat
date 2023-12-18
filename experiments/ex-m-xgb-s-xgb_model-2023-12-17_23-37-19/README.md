
# Experiment: m-xgb-s-xgb_model

## Parameters
- Historical data starting from: 2021-10-01 00:00:00
- Historical data ending at: 2021-10-10 00:00:00
- Streaming data starting from: 2021-10-10 00:00:00
- Streaming data ending at: 2022-11-01 00:00:00
- Time interval: 60

## Results
### Model Performance Metrics
| Model                               | MAE (s)   | MAPE (%)   | RMSE (s)   |
|-------------------------------------|-----------|------------|------------|
| Base Model (XGBoost)                 | 10.973392029746565 | 58.004968550866124 | 18.700606886616313 |
| Base Model with Incremental Learning | 2928715412.5245      | 20036907492.68914      | 22103375026.389137      |

    