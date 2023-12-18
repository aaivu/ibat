
# Experiment: m-xgb-s-xgb_model

## Parameters
- Historical data starting from: 2021-10-01 00:00:00
- Historical data ending at: 2021-10-10 00:00:00
- Streaming data starting from: 2021-10-10 00:00:00
- Streaming data ending at: 2022-01-01 00:00:00
- Time interval: 15

## Results
### Model Performance Metrics
| Model                               | MAE (s)   | MAPE (%)   | RMSE (s)   |
|-------------------------------------|-----------|------------|------------|
| Base Model (XGBoost)                 | 11.071693928768983 | 52.397146526929994 | 20.18103218913551 |
| Base Model with Incremental Learning | 6689832.141383576      | 34550561.301537745      | 41190215.9537504      |

    