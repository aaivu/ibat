import warnings
from typing import Optional, List

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, to_datetime
from pandas.errors import SettingWithCopyWarning
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from src.datasets import (
    BUS_654_FEATURES_ADDED_DWELL_TIMES,
    BUS_654_FEATURES_ADDED_RUNNING_TIMES,
)
from src.models.use_cases.arrival_time.bus import MME4BAT



def plot_mean_dwell_time(bus_stop_value, start, end, df, folder_path):

    df_dwell_time_in_seconds = df.groupby('date')['dwell_time_in_seconds'].mean().reset_index()
    df_dwell_time_in_seconds = df_dwell_time_in_seconds.sort_values(by='date')
    df_base_model_prediction = df.groupby('date')['base_model_prediction'].mean().reset_index()
    df_base_model_prediction = df_base_model_prediction.sort_values(by='date')
    df_model_prediction = df.groupby('date')['model_prediction'].mean().reset_index()
    df_model_prediction = df_model_prediction.sort_values(by='date')

    plt.figure(figsize=(13,5))

    # x = list(range(len(df['true_prediction'])))
    x = df_model_prediction['date']

    plt.plot(x, df_dwell_time_in_seconds['dwell_time_in_seconds'], label="True dwell time", marker='o', linestyle='-')
    plt.plot(x, df_base_model_prediction['base_model_prediction'], label="Base model prediction", marker='o', linestyle='--')
    plt.plot(x, df_model_prediction['model_prediction'], label="Prediction with incremental online learning", marker='o', linestyle='--')

    plt.xlabel('Date')
    plt.ylabel('Mean Dwell Time')
    plt.title(f'Mean dwell time at the bus stop {bus_stop_value} on each day at [{start} - {end}) h')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    file_name = f'd{bus_stop_value}.png'  
    save_path = f'{folder_path}/{file_name}'
    plt.savefig(save_path,dpi=200)
    # plt.show()
    return None



def run() -> None:
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    rt_df: DataFrame = BUS_654_FEATURES_ADDED_RUNNING_TIMES.dataframe
    dt_df: DataFrame = BUS_654_FEATURES_ADDED_DWELL_TIMES.dataframe

    numerical_rt_df = rt_df.select_dtypes(include='number')
    numerical_dt_df = dt_df.select_dtypes(include='number')

    base_model: Optional[MME4BAT] = None
    model: Optional[MME4BAT] = None

    true_prediction = []
    base_model_prediction = []
    model_prediction = []

    numerical_dt_df = numerical_dt_df[:10000]
    dt_df = dt_df[:10000]
    from_idx = 0

    for to_idx in range(1000, len(numerical_dt_df), 100):
        dt_chunk: DataFrame = numerical_dt_df.iloc[from_idx:to_idx, :].reset_index(drop=True)
        dt_x: DataFrame = dt_chunk.drop(columns=["dwell_time_in_seconds"])
        dt_y: DataFrame = dt_chunk[["dwell_time_in_seconds"]]

        if not model:
            base_model = MME4BAT()
            model = MME4BAT()

            base_model.fit(rt_x=None, rt_y=None, dt_x=dt_x, dt_y=dt_y)
            model.fit(rt_x=None, rt_y=None, dt_x=dt_x, dt_y=dt_y)
        else:
            true_prediction.extend(dt_y["dwell_time_in_seconds"].tolist())
            base_model_prediction.extend(
                base_model.predict(rt_x=None, dt_x=dt_x)["prediction"].tolist()
            )
            model_prediction.extend(
                model.predict(rt_x=None, dt_x=dt_x)["prediction"].tolist()
            )

            model.incremental_fit(
                ni_rt_x=None, ni_rt_y=None, ni_dt_x=dt_x, ni_dt_y=dt_y
            )

        # dt_next_chunk = numerical_dt_df.iloc[to_idx: (to_idx + 10), :]
        # dt_x = dt_next_chunk[["deviceid", "bus_stop", "day_of_week", "time_of_day"]]
        # dt_y = dt_next_chunk[["dwell_time_in_seconds"]]

        from_idx = to_idx

    print(
        f"MAE for base model: {mean_absolute_error(true_prediction, base_model_prediction)}"
    )
    print(
        f"MAPE for base model: {mean_absolute_percentage_error(true_prediction, base_model_prediction) * 100}"
    )
    print(
        f"RMSE for base model: {mean_squared_error(true_prediction, base_model_prediction, squared=False)}"
    )
    print("-----------------------------------------------------------------")
    print(
        f"MAE for incremental online learning model: {mean_absolute_error(true_prediction, model_prediction)}"
    )
    print(
        f"""MAPE for incremental online learning model: {
        mean_absolute_percentage_error(true_prediction, model_prediction) * 100
    }"""
    )
    print(
        f"""RMSE for incremental online learning model: {
        mean_squared_error(true_prediction, model_prediction, squared=False)
    }"""
    )

    dt_df_new = dt_df[1000:-100]
    if len(dt_df_new) == len(true_prediction) == len(model_prediction) == len(base_model_prediction):
        dt_df_new['true_prediction'] = true_prediction
        dt_df_new['model_prediction'] = model_prediction
        dt_df_new['base_model_prediction'] = base_model_prediction
    else:
        print(len(true_prediction), len(model_prediction), len(base_model_prediction), dt_df_new.shape)
        print("Error: The lengths of the arrays and the DataFrame do not match.")


    plt.figure(figsize=(100, 20))

    x = list(range(len(dt_df_new['true_prediction'])))
    plt.plot(x, dt_df_new['true_prediction'], label="True dwell time")
    plt.plot(x, dt_df_new['base_model_prediction'], label="Base model prediction")
    plt.plot(x, dt_df_new['model_prediction'], label="Prediction with incremental online learning")

    plt.xlabel("X")
    plt.ylabel("Dwell time (s)")
    plt.title("Multiple Lines on a Common Graph")

    plt.legend()
    # plt.show()
    plt.savefig("dt.png", dpi=150)





    # Task 

    direction_value = 1 
    bus_stop_value  = 103
    time_of_day_value = 10
    start  = '10:00:00'
    end = '10:15:00'
    img_path = "./"

    df = dt_df_new
    df['arrival_time'] = pd.to_datetime(df['arrival_time'])
    df['time_of_day'] = df['arrival_time'].dt.hour

    bus_stop_list = df['bus_stop'].unique()
    for bus_stop_value in bus_stop_list:          
        filtered_df = df[(df['direction'] == direction_value) & (df['bus_stop'] == bus_stop_value) & (df['time_of_day'] == time_of_day_value)]
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        filtered_df['arrival_time'] = pd.to_datetime(filtered_df['arrival_time'])

        start_time = pd.Timestamp(start).time()
        end_time = pd.Timestamp(end).time()

        filtered_df = filtered_df[(filtered_df['arrival_time'].dt.time >= start_time) & (filtered_df['arrival_time'].dt.time <= end_time)]
        print(bus_stop_value , filtered_df.shape)
        plot_mean_dwell_time(bus_stop_value,start,end, filtered_df, img_path)