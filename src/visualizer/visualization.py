import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec




def plot_mean_dwell_time(bus_stop_value, start, end, df_x , df_y, folder_path):
    plt.figure(figsize=(13,5))
    plt.plot(df_x, df_y, marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Mean Dwell Time')
    plt.title(f'Mean dwell time at the bus stop {bus_stop_value} on each day at [{start} - {end}) h')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    file_name = f'd{bus_stop_value}.png'  
    save_path = f'{folder_path}/{file_name}'
    plt.savefig(save_path,dpi=200)
    plt.show()
    return None

def plot_mean_arrival_time(bus_stop_value, start, end, df_x , df_y, folder_path):
    plt.figure(figsize=(13,5))
    plt.plot(df_x, df_y, marker='o', linestyle='-')
    myFmt = mdates.DateFormatter('%H:%M:%S') 
    plt.gca().yaxis.set_major_formatter(myFmt)
    plt.xlabel('Date')
    plt.ylabel('Mean Arrival Time')
    plt.title(f'Mean arrival time at the bus stop {bus_stop_value} on each day at [{start} - {end}) h')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    file_name = f'a{bus_stop_value}.png'  
    save_path = f'{folder_path}/{file_name}'
    plt.savefig(save_path,dpi=200)
    plt.show()
    return None

def plot_data_drift_detections(stream, bus_stop, drifts = None):
    fig = plt.figure(figsize=(7,3), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    plt.grid()
    plt.plot(stream, label='Stream Data Drift Detection')
    if drifts is not None:
        for drift_detected in drifts:
            plt.axvline(drift_detected, color='red')
    plt.ylabel(f'mean arrival time for {bus_stop} ')
    plt.show()
    return None

    

# TEST CODE

direction_value = 1 
bus_stop_value  = 102 
time_of_day_value = 10
start  = '10:00:00'
end = '10:15:00'
csv_path = "src/datasets/_datasets/bus_dwell_times_654.csv"
img_path = "./"

df = pd.read_csv(csv_path)

df['arrival_time'] = pd.to_datetime(df['arrival_time'])
df['time_of_day'] = df['arrival_time'].dt.hour
    
filtered_df = df[(df['direction'] == direction_value) & (df['bus_stop'] == bus_stop_value) & (df['time_of_day'] == time_of_day_value)]

filtered_df['date'] = pd.to_datetime(filtered_df['date'])

filtered_df['arrival_time'] = pd.to_datetime(filtered_df['arrival_time'])

start_time = pd.Timestamp(start).time()
end_time = pd.Timestamp(end).time()

filtered_df = filtered_df[(filtered_df['arrival_time'].dt.time >= start_time) & (filtered_df['arrival_time'].dt.time <= end_time)]



mean_dwell_times = filtered_df.groupby('date')['dwell_time_in_seconds'].mean().reset_index()
sorted_mean_dwell_times = mean_dwell_times.sort_values(by='date')
print(sorted_mean_dwell_times.head())

df_x = sorted_mean_dwell_times['date']
df_y = sorted_mean_dwell_times['dwell_time_in_seconds']
plot_mean_dwell_time(bus_stop_value,start,end,df_x , df_y, img_path)

mean_arrival_times = filtered_df.groupby('date')['arrival_time'].mean().reset_index()
sorted_mean_arrival_times = mean_arrival_times.sort_values(by='date')
print(sorted_mean_arrival_times.head())

df_x = sorted_mean_arrival_times['date']
df_y = sorted_mean_arrival_times['arrival_time']
plot_mean_arrival_time(bus_stop_value,start,end,df_x , df_y, img_path)


