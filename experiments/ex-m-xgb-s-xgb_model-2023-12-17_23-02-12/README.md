
# Experiment: m-xgb-s-xgb_model

## Parameters
- Historical data starting from: 2021-10-01 00:00:00
- Historical data ending at: 2021-10-10 00:00:00
- Streaming data starting from: 2021-10-10 00:00:00
- Streaming data ending at: 2021-11-01 00:00:00
- Time interval: 60

## Results
### Model Performance Metrics
| Model                               | MAE (s)   | MAPE (%)   | RMSE (s)   |
|-------------------------------------|-----------|------------|------------|
| Base Model (XGBoost)                 | 11.031494904113444 | 50.88909107177862 | 20.646408296378063 |
| Base Model with Incremental Learning | 4.367322845307889      | 19.693678365577217      | 7.577372996552075      |

    