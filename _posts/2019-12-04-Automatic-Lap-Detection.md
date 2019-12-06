---
layout: post
title: "Automatic lap detection in GPS files"
tags: [data-science, fourier-analysis, dtw, gps]
author: jelledb
---
In this post we try to automatically detect the number of laps and align them from GPS files of riders of a same race (which contains a list of timestamped coordinates).

#### We first install and import some dependencies


```python
# We will first use fitfiles (and install the library fitparse) + we'll also install some dependencies for further analysis and plotting
!pip install fitparse
!pip install pandas
!pip install plotnine

# Data processing libraries
import pandas as pd
import numpy as np

# Plotting libraries
from plotnine import *
import plotnine

import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import copy

# Tweaking matplotlib and warnings
mpl.rcParams["figure.figsize"] = "30, 8"
warnings.filterwarnings("ignore")

%matplotlib inline
```

#### Code to process workout files into workout records containing all the fields from a common GPS file


```python
#@title The code to process the .fit files
from fitparse import FitFile
import datetime
import sys
import json
import time
import pandas as pd
from dateutil import tz


def getFitDict(file_path):
    print(file_path)
    fitfile = FitFile(file_path)
    # Get summary of fit file
    summary = {}
    for activity in fitfile.get_messages('session'):
        for record_data in activity:
            value = record_data.value
            summary[record_data.name]=value

    # Get all data messages that are of type record
    records=[]
    for record in fitfile.get_messages('record'):

        # Go through all the data entries in this record
        activities=[]
        activity_line={}
        for record_data in record:
            value = record_data.value
            
            if isinstance(value, datetime.datetime):
                value = record_data.value.strftime("%Y-%m-%d %H:%M:%S")
            if (record_data.name is 'position_lat' or record_data.name is 'position_long'):
                value = record_data.value * (180./2**31)
                
            activity_line[record_data.name] = value	
        records.append(activity_line)

    gear_changes = []
    for event in fitfile.get_messages('event'):
        #Here can be shifting data
        event_line = {}
        for event_data in event:
            value = event_data.value
            event_line[event_data.name] = value
        if event_line["event"] in ["rear_gear_change", "front_gear_change"]:
            gear_changes.append(event_line)


    # Make it work for Python 2+3 and with Unicode
    import io
    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str
        
    # Get all data messages that are of type lap
    laps=[]
    for record in fitfile.get_messages('lap'):

        # Go through all the data entries in this lap
        activities=[]
        activity_line={}
        for record_data in record:
            value = record_data.value
            
            if isinstance(value, datetime.datetime):
                value = record_data.value.strftime("%Y-%m-%d %H:%M:%S")
            if (record_data.name is 'position_lat' or record_data.name is 'position_long' or
            record_data.name is 'start_position_lat' or record_data.name is 'start_position_long' or
            record_data.name is 'end_position_lat' or record_data.name is 'end_position_long'):
                value =  record_data.value * (180./2**31)
                
            activity_line[record_data.name] = value
        laps.append(activity_line)

    # Write JSON file
    output=summary
    output['records']=records
    output['laps']=laps
    output['gear_changes']=gear_changes

    return output

def summarize_shifting(stats, workout):
    time_in_gears = {}
    total_shifts = {"front": 0, "rear": 0}

    start_timestamp = workout["start_time"]
    print(start_timestamp)
    previous_timestamp = start_timestamp
    previous_gear = "unknown"

    for shift in stats['gear_changes']:
        current_timestamp = shift['timestamp']
        print(current_timestamp)
        elapsed_time = (current_timestamp-previous_timestamp).seconds

        shift_type = shift['event']
        if shift_type == 'rear_gear_change':
            total_shifts["rear"]+=1
        elif shift_type == 'front_gear_change':
            total_shifts["front"]+=1

        gear_combo = str(shift["front_gear_num"])+"x"+str(shift["rear_gear_num"])
        
        time_in_gears[previous_gear] = time_in_gears.get(previous_gear, 0)+elapsed_time
        
        previous_gear = gear_combo
        previous_timestamp = current_timestamp

    return {"time_in_gears": time_in_gears,
            "total_shifts": total_shifts}
            
def get_sensor_data(records, data_type):
    output = []
    start_time = datetime.datetime.strptime(records[0]['timestamp'], "%Y-%m-%d %H:%M:%S")
    if data_type in ['altitude', 'cadence',  'power',
                        'heart_rate', 'temperature', 'timestamp',
                        'left_right_balance', 'altitude', 'acceleration']:
        for record in records:
            if data_type == 'power' and 'power' in record:
                output_record = {
                                'elapsed_time': int((datetime.datetime.strptime(record['timestamp'],"%Y-%m-%d %H:%M:%S")-start_time).total_seconds()),
                                'power': record['power']
                }
                if 'left_pedal_smoothness' in record:
                    output_record['left_pedal_smoothness']=record['left_pedal_smoothness']
                if 'right_pedal_smoothness' in record:
                    output_record['right_pedal_smoothness']=record['right_pedal_smoothness']
                if 'left_torque_effectiveness' in record:
                    output_record['left_torque_effectiveness']=record['left_torque_effectiveness']
                if 'right_torque_effectiveness' in record:
                    output_record['right_torque_effectiveness']=record['right_torque_effectiveness']
                if 'left_right_balance' in record:
                    if record['left_right_balance'] is not None:
                        output_record['left_right_balance']=record['left_right_balance']
                        output_record['left_power']=record['left_right_balance']/100.0*record['power']
                        output_record['right_power']=(100-record['left_right_balance'])/100.0*record['power']
                    elif record['left_right_balance'] is None:
                        output_record['left_right_balance']=0
                        output_record['left_power']=0
                        output_record['right_power']=0
                output.append(output_record)
            elif data_type in record:
                output.append({
                                'elapsed_time': int((datetime.datetime.strptime(record['timestamp'],"%Y-%m-%d %H:%M:%S")-start_time).total_seconds()),
                                data_type: record[data_type]
                })
    elif data_type == "coordinates":
         for record in records:
            if "position_lat" in record and "position_long" in record:
                output.append({
                                'elapsed_time': int((datetime.datetime.strptime(record['timestamp'],"%Y-%m-%d %H:%M:%S")-start_time).total_seconds()),
                                data_type: [record['position_lat'], record['position_long']]
                })
    return output
        
def add_accelerometer_data(workout, workout_data, file):
    acc_df = pd.read_csv(file, sep="\t")
    acc_df['TStamp Europe/Brussels'] = pd.to_datetime(acc_df['TStamp Europe/Brussels']/1000,unit='s')
        
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    activity_started_at = workout['start_time'].replace(tzinfo=from_zone).astimezone(to_zone).replace(tzinfo=None)
    activity_ended_at = workout['timestamp'].replace(tzinfo=from_zone).astimezone(to_zone).replace(tzinfo=None)

    acc_df.columns = ['datetime','X','Y','Z']

    mask = (acc_df['datetime']>activity_started_at) & (acc_df['datetime']<=activity_ended_at)
    acc_df = acc_df.loc[mask]

    acc_df.set_index('datetime', inplace=True)
    acc_df = acc_df.resample('1s').mean()

    print("Filtering out accelerometer data between {} and {}.".format(activity_started_at, activity_ended_at))
    df_counter = 0
    print("first accelerometer data timestamp: {}".format(acc_df.index[0]))
    for i in range(0, len(workout_data['records'])):
        record_time = datetime.datetime.strptime(workout_data['records'][i]['timestamp'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=from_zone).astimezone(to_zone).replace(tzinfo=None)
        while acc_df.index[df_counter]<record_time:
            df_counter+=1
        workout_data['records'][i]['acceleration']={
            'x': acc_df.iloc[df_counter]['X'],
            'y': acc_df.iloc[df_counter]['Y'],
            'z': acc_df.iloc[df_counter]['Z']
        }

    return workout_data

def fit_to_json(file_path):
    print(file_path)
    fitfile = FitFile(file_path)

    # Get all data messages that are of type record
    records=[]
    for record in fitfile.get_messages('record'):

        # Go through all the data entries in this record
        activities=[]
        activity_line={}
        for record_data in record:
            value = record_data.value
            
            if isinstance(value, datetime.datetime):
                value = record_data.value.strftime("%Y-%m-%d %H:%M:%S")
            if (record_data.name is 'position_lat' or record_data.name is 'position_long'):
                value = record_data.value * (180./2**31)
                
            activity_line[record_data.name] = value	
        records.append(activity_line)

    # Make it work for Python 2+3 and with Unicode
    import io
    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str
        
    # Get all data messages that are of type lap
    laps=[]
    for record in fitfile.get_messages('lap'):

        # Go through all the data entries in this lap
        activities=[]
        activity_line={}
        for record_data in record:
            value = record_data.value
            
            if isinstance(value, datetime.datetime):
                value = record_data.value.strftime("%Y-%m-%d %H:%M:%S")
            if (record_data.name is 'position_lat' or record_data.name is 'position_long' or
            record_data.name is 'start_position_lat' or record_data.name is 'start_position_long' or
            record_data.name is 'end_position_lat' or record_data.name is 'end_position_long'):
                value = record_data.value * (180./2**31)
                
            activity_line[record_data.name] = value
        laps.append(activity_line)

    # Make it work for Python 2+3 and with Unicode
    import io
    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    # Write JSON file
    output={'records':records, 'laps':laps}

    return json.dumps(output)
    
def fields_to_array(records, desired_fields):
    output = []
    for record in records:
        row = []
        for field in desired_fields:
            row.append(record[field])
        output.append(row)
    return output
```

## Upload a workout file to this Notebook
Currently supported are:
- **fit files**: using the fitparse library
- **csv files**: which are result of own Strava activity parsing


```python
#@title Upload (a) .fit/.csv file(s)
from google.colab import files

uploaded = files.upload()
SUPPORTED_FILE_TYPES = ['fit','csv']
file_names = []

for fn in uploaded.keys():
    while fn.split('.')[-1] not in SUPPORTED_FILE_TYPES:
        print("The file is a .{} file, only {} files are supported.".format(fn.split(".")[-1], ",".join(SUPPORTED_FILE_TYPES)))
        !rm $fn
        uploaded = files.upload()
        fn = next(iter(uploaded.keys()))
    print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
    file_names.append(fn)
```



     <input type="file" id="files-2049d5aa-4ef7-44c6-b49c-75f20488e1e4" name="files[]" multiple disabled />
     <output id="result-2049d5aa-4ef7-44c6-b49c-75f20488e1e4">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving 38467BED.fit to 38467BED.fit
    User uploaded file "38467BED.fit" with length 161869 bytes


**Next step:** convert the records of the gps files into a list of records (with fields such as latitude, longitude, power, ...)


```python
workout_records = []

for file_name in file_names:
    if file_name.endswith('fit'):
        workout_record = getFitDict(file_name)['records']
    elif file_name.endswith('csv'):
        workout_record = pd.read_csv(file_name, sep=',').to_dict('r')
    workout_records.append(workout_record)
```

    38467BED.fit


Print the available fields in the first file (if you uploaded multiple ones)


```python
i = 0
for file_name in file_names:
    print(file_name)
    print("Available fields. ")
    print(",".join(workout_records[i][0].keys()))
    i+=1
```

    38467BED.fit
    Available fields. 
    accumulated_power,altitude,cadence,calories,distance,enhanced_altitude,enhanced_speed,grade,heart_rate,left_pedal_smoothness,left_right_balance,left_torque_effectiveness,position_lat,position_long,power,right_pedal_smoothness,right_torque_effectiveness,speed,temperature,timestamp



```python
workout_dfs = []
```


```python
#@title Add timestamp and lat, lon to dataframe and calculate elapsed time in seconds
counter = 0
for file_name in file_names:
    records = workout_records[counter]
    if file_name.endswith('fit'):
        fields_to_analyze = np.array(fields_to_array(records, ['timestamp','position_lat','position_long']))
    elif file_name.endswith('csv'):
        fields_to_analyze = np.array(fields_to_array(records, ['timestamp','latitude','longitude']))

    workout_df = pd.DataFrame(data=fields_to_analyze[:,1:],    # values
                            index=fields_to_analyze[:,0],    # 1st column as index
                                columns=['latitude','longitude'])
    workout_df.index = pd.to_datetime(workout_df.index)
    start_time = workout_df.index[0]
    workout_df['latitude']=workout_df['latitude'].astype(float)
    workout_df['longitude']=workout_df['longitude'].astype(float)
    workout_df = workout_df.resample('s', how='max').interpolate(method='linear')
    workout_df['Datetime'] = workout_df.index
    start_time = workout_df.Datetime[0]
    print(start_time)
    workout_df['Elapsed_time']=pd.Series(workout_df.Datetime-start_time).dt.total_seconds()

    workout_dfs.append(workout_df)
    counter+=1
```

    2019-12-01 14:02:22


A converted dataframe now looks like this


```python
workout_dfs[0].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>latitude</th>
      <th>longitude</th>
      <th>Datetime</th>
      <th>Elapsed_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-12-01 14:02:22</th>
      <td>51.237713</td>
      <td>3.394923</td>
      <td>2019-12-01 14:02:22</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2019-12-01 14:02:23</th>
      <td>51.237716</td>
      <td>3.394930</td>
      <td>2019-12-01 14:02:23</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2019-12-01 14:02:24</th>
      <td>51.237716</td>
      <td>3.394935</td>
      <td>2019-12-01 14:02:24</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2019-12-01 14:02:25</th>
      <td>51.237723</td>
      <td>3.394942</td>
      <td>2019-12-01 14:02:25</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2019-12-01 14:02:26</th>
      <td>51.237724</td>
      <td>3.394940</td>
      <td>2019-12-01 14:02:26</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2019-12-01 14:02:27</th>
      <td>51.237726</td>
      <td>3.394938</td>
      <td>2019-12-01 14:02:27</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2019-12-01 14:02:28</th>
      <td>51.237728</td>
      <td>3.394938</td>
      <td>2019-12-01 14:02:28</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2019-12-01 14:02:29</th>
      <td>51.237730</td>
      <td>3.394933</td>
      <td>2019-12-01 14:02:29</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2019-12-01 14:02:30</th>
      <td>51.237730</td>
      <td>3.394932</td>
      <td>2019-12-01 14:02:30</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2019-12-01 14:02:31</th>
      <td>51.237730</td>
      <td>3.394935</td>
      <td>2019-12-01 14:02:31</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>



Showing the filename of the first dataframe


```python
workout_df = workout_dfs[0]
rider_name = file_names[0].split(".")[-2].split("_")[-1]
print(rider_name)
```

    38467BED



```python
#@title Plot latitude over time to find patterns
plotnine.options.figure_size = (20, 4.8)
(
    ggplot(workout_df)+
    aes(x='Elapsed_time', y='latitude')+
    geom_line(color='red')+
    labs(title='Latitude over elapsed time', x='Elapsed Time (seconds)', y='Latitude (degrees)')
)
```


![png](/assets/img/posts/Automatic_Lap_Detection_17_0.png)





    <ggplot: (8733914093369)>




```python
#@title Same for longitude
plotnine.options.figure_size = (20, 4.8)
(
    ggplot(workout_df)+
    aes(x='Elapsed_time', y='longitude')+
    geom_line(color='blue')+
    labs(title='Longitude over elapsed time', x='Elapsed Time (seconds)', y='Longitude (degrees)')
)
```


![png](/assets/img/posts/Automatic_Lap_Detection_18_0.png)





    <ggplot: (8733914045988)>




```python
#@title FFT analysis functions run to implement
from scipy.fftpack import fft
from scipy import signal
import scipy as sp

def do_fourier_analysis(workout_df, field_name):
    N = len(workout_df[field_name])
    fs = 1 #Gps files are recorded on 1hz

    temp_fft = sp.fftpack.fft(workout_df[field_name])
    temp_psd = np.abs(temp_fft) ** 2
    fftfreq = sp.fftpack.fftfreq(len(temp_psd), 1. / N)

    i = fftfreq > 0

    return fftfreq[i], temp_psd[i]

def plot_spectrum(workout_df, field_name):
    fftfreq, temp_psd = do_fourier_analysis(workout_df, field_name)
    fig, ax = plt.subplots(1, 1, figsize=(20, 4))
    ax.bar(fftfreq, temp_psd)
    max_x = (np.argmax(temp_psd)+1)*2 + 1
    ax.set_xlim(0, max_x)
    ax.set_xticks(np.arange(0,max_x,1))
    ax.set_xlabel('Frequency (laps in file)')
    ax.set_ylabel('Power spectrum')
    ax.set_title("Power spectrum based on "+ field_name+ " of number of laps in file")

def get_number_of_laps(workout_df, field_name):
    fftfreq, temp_psd = do_fourier_analysis(workout_df, field_name)
    laps = (np.argmax(temp_psd)+1)

    return laps
```


```python
plot_spectrum(workout_df, 'latitude')
```


![png](/assets/img/posts/Automatic_Lap_Detection_20_0.png)



```python
plot_spectrum(workout_df, 'longitude')
```


![png](/assets/img/posts/Automatic_Lap_Detection_21_0.png)


Show the "best guess" about the number of laps using Fourier Transform


```python
number_of_laps = get_number_of_laps(workout_df, 'longitude') #We don't include 0 laps, so we have to add 1 afterwards
print("We think you completed ", number_of_laps, " laps.")
```

    We think you completed  10  laps.


Now we'll try to detect the lap boundaries based on the knowledge of the number of laps and a "generous threshold" around the candidate lap splits. The following functions help to do do this exact job.


```python
# Find matching points
import sys
import math

THRESHOLD = 15  # percentage before or after the "perfect next lap point position"
                # - "perfect next lap point position" = point_index + (number_of_points/total_laps)
                # - We set a margin of a certain percentage to get an array of possible "candidate points for the match"
                
def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d*1000

def find_closest_match(workout_df, point_index, number_of_laps, max_radius_m=1, first_lap=False):
    number_of_points = len(workout_df)
    theoretical_next_point = int(point_index+number_of_points/number_of_laps)
    fraction=1/100 if first_lap else 15/100
    margin = int(fraction*number_of_points/number_of_laps)
    next_point_range = np.arange(theoretical_next_point-margin, theoretical_next_point+margin, 1)
    next_point_range = next_point_range[next_point_range<number_of_points]
    origin_point = (workout_df.iloc[point_index]['latitude'], workout_df.iloc[point_index]['longitude'])
    min_distance = sys.maxsize
    min_index = -1
    for j in next_point_range:
        candidate_point = (workout_df.iloc[j]['latitude'], workout_df.iloc[j]['longitude'])
        distance_between = distance(origin_point, candidate_point)
        if distance_between<=min_distance:
            min_distance = distance_between
            min_index = j
    if min_distance<max_radius_m:
        return min_index, min_distance
    else:
        return None, None
```

Now detect the lap split points based on a workout df and the number of lap. For the first lap we put the range of possible next points a bit more strict to get the most accurate match possible.


```python
first_lap_candidates = []
latitudes = workout_df['latitude'].tolist()
longitudes = workout_df['longitude'].tolist()

def find_next_lap(start_point, workout_df, number_of_laps, first_lap=False):
    found = False
    i=start_point
    lap_bounds = None

    while not found and i<len(workout_df):
        if first_lap:
            next_point, accuracy = find_closest_match(workout_df, i, number_of_laps, first_lap=True)
        else:
            next_point, accuracy = find_closest_match(workout_df, i, number_of_laps, max_radius_m=10, first_lap=False)
        if next_point:
            lap_bounds = [i,next_point]
            found=True
        else:
            i+=1 
  
    return lap_bounds

laps = []
stop = False
i=0
lap_start = 0
while not stop and lap_start < len(workout_df):
    lap = find_next_lap(lap_start, workout_df, number_of_laps, first_lap=(i==0))
    if lap:
        laps.append(lap)
        lap_start = lap[1]
        i+=1
    else:
        stop = True

print("Print possible laps are : ", laps)
```

    Print possible laps are :  [[48, 460], [460, 844], [844, 1264], [1264, 1625], [1625, 2015], [2015, 2405], [2405, 2834], [2834, 3193], [3193, 3588]]


Visualy show the detected lap split points, they seem to look quite nicely representing the laps (consequently)


```python
plotnine.options.figure_size = (20, 4.8)
plot = (
    ggplot(workout_df)+
    aes(x='Elapsed_time', y='longitude')+
    geom_line(color='blue')+
    labs(title='Longitude over elapsed time of {}'.format(rider_name), x='Elapsed Time (seconds)', y='longitude (degrees)')
)

for lap in laps:
    plot+=geom_vline(xintercept = lap[0], linetype="dotted", color = "red", size=1.5)
    plot+=geom_vline(xintercept = lap[1], linetype="dotted", color = "red", size=1.5)

plot
```


![png](/assets/img/posts/Automatic_Lap_Detection_29_0.png)





    <ggplot: (-9223363302943166439)>



#### Next step: trying to align the different laps with each other

We'll use Dynamic Time Warping for this job


```python
!pip install fastdtw
from fastdtw import fastdtw
```

    Requirement already satisfied: fastdtw in /usr/local/lib/python3.6/dist-packages (0.3.4)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from fastdtw) (1.17.4)


Comparison function for the DTW algorithm. We'll be using Euclidean distance, as it makes sense to use this metric for geospatial analysis.


```python
def compare(lap1, lap2):
    haversine_dist = lambda x, y: distance((x[0],x[1]),(y[0],y[1]))

    d, path = fastdtw(lap1, lap2, dist=haversine_dist)

    return d, path
```


```python
def plot_wrap(path, lap, series1, series2, label1, label2, series_description):
    # You can also visualise the accumulated cost and the shortest path
    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = 20, 20
    lap_1 = np.arange(lap[0], lap[1], 1)
    plt.plot(lap_1, series1[lap[0]:lap[1]], color='r', label=label1)
    print(lap_1)
    plt.ylabel(series_description)
    i=0
    start_l2 = 0
    while path[i][0]<lap[0]:
        i+=1
    start_l2 = path[i][1]
    while path[i][0]<lap[1]:
        match = path[i]
        #print("[{}, {}]".format(match[0], match[1]))
        #print("[{}, {}]".format(series1[match[0]], series2[match[1]]))
        x_points = [match[0], match[1]]
        y_points = [series1[match[0]], series2[match[1]]]

        plt.plot(x_points, y_points, 'o-', color='b', linewidth=0.1)
        i+=1
    end_l2 = path[i][1]
    lap_2 = np.arange(start_l2, end_l2, 1)
    plt.plot(lap_2, series2[lap_2], color='g',label=label2)

    plt.show()

def plot_wrap_laps(path, laps, lap_index_1, lap_index_2, series, series_description):
    # You can also visualise the accumulated cost and the shortest path
    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = 20, 20
    lap = laps[lap_index_1]
    lap_1 = np.arange(lap[0], lap[1], 1)
    plt.plot(lap_1, series[lap[0]:lap[1]], color='r', label="Lap {}".format(lap_index_1+1))
    plt.title(series_description + " Lap {} vs Lap {}".format(lap_index_1+1, lap_index_2+1))
    i=0
    start_l2 =  path[0][1]+laps[lap_index_2][0]
    while i<len(path):
        match = path[i]
        x_points = [match[0]+laps[lap_index_1][0], match[1]+laps[lap_index_2][0]]
        y_points = [series[match[0]+laps[lap_index_1][0]], series[match[1]+laps[lap_index_2][0]]]

        plt.plot(x_points, y_points, 'o-', color='b', linewidth=0.1)
        i+=1
    end_l2 = path[i-1][1]+laps[lap_index_2][0]
    lap_2 = np.arange(start_l2, end_l2, 1)
    plt.plot(lap_2, series[lap_2], color='g',label="Lap {}".format(lap_index_2+1))

    plt.show()
```


```python
#@title
import numpy as np

# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x
first_lap = np.array(workout_df.iloc[laps[0][0]:laps[0][1]][['latitude','longitude']].values)
second_lap = np.array(workout_df.iloc[laps[1][0]:laps[1][1]][['latitude','longitude']].values)
third_lap = np.array(workout_df.iloc[laps[2][0]:laps[2][1]][['latitude','longitude']].values)
fourth_lap = np.array(workout_df.iloc[laps[3][0]:laps[3][1]][['latitude','longitude']].values)
fifth_lap = np.array(workout_df.iloc[laps[4][0]:laps[4][1]][['latitude','longitude']].values)
sixth_lap = np.array(workout_df.iloc[laps[5][0]:laps[5][1]][['latitude','longitude']].values)

def compare_laps(workout_df, laps, first, second):
    lap1 = np.array(workout_df.iloc[laps[first-1][0]:laps[first-1][1]][['latitude','longitude']].values)
    lap2 = np.array(workout_df.iloc[laps[second-1][0]:laps[second-1][1]][['latitude','longitude']].values)

    d, path = compare(lap1, lap2)

    return d, path
    
def get_lap(workout_df, laps, lap_number):
    return np.array(workout_df.iloc[laps[lap_number-1][0]:laps[lap_number-1][1]][['latitude','longitude']].values)

def plot_comparison(workout_df, laps, first, second, title):
    plot_wrap_laps(path, laps, first-1, second-1, np.array(workout_df[['longitude']].values), title)

def get_matching_accuracies(workout_df, laps):
    accuracy_matrix = []
    for i in range(0, len(laps)):
        accuracies = []
        for j in range(0, i):
            accuracies.append(-1)
        accuracies.append(0)
        for j in range(i, len(laps)):
            if i<j:
                d, path = compare_laps(workout_df, laps, i+1, j+1)
                accuracies.append(d/len(path))
        accuracy_matrix.append(accuracies)
    return accuracy_matrix
```


```python
import seaborn as sns; sns.set()

accuracy = get_matching_accuracies(workout_dfs[0], laps)

#Which laps are matching the best, darker color --> better match (low dtw distance) 
ax = sns.heatmap(np.array(accuracy), cmap='pink')
ax.set_title("Heat map of DTW distance/point between laps (lower distance is darker color)")
ax.set_xticklabels(np.arange(1, number_of_laps))
ax.set_yticklabels(np.arange(1, number_of_laps))
ax
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1859da2be0>




![png](/assets/img/posts/Automatic_Lap_Detection_38_1.png)


Visualize the DTW wrap of laps of the same rider / or of different riders


```python
# If you have multiple dataframes (different riders at same race)
# hermans = np.array(workout_dfs[3][['latitude','longitude']].values)
# debock = np.array(workout_dfs[0][['latitude','longitude']].values)

lap_number_1 = 6
lap_number_2 = 9

# We'll now compare 2 laps from the same rider
d, path =  compare_laps(workout_dfs[0], laps, lap_number_1, lap_number_2)

# The DTW spacing between the two curves (Euclidean distance) in meter
# Note that the GPS accuracy is +-4m (as stated by the US government)
#   We can get an idea of confidence when dividing the sum of differences divided through the number of wrapped points
print("Total DTW distance: ", d, "m")
print("Averaged by wrapped path: ", d/len(path), "m")

# Plot a graph showing the alignment between the laps
plot_comparison(workout_dfs[0], laps, lap_number_1, lap_number_2, "Moerkerke (Jelle De Bock) : lap {} vs lap {}".format(lap_number_1,lap_number_2))
```

    Total DTW distance:  1463.4862749221459 m
    Averaged by wrapped path:  3.5264729516196285 m



![png](/assets/img/posts/Automatic_Lap_Detection_40_1.png)



```python
#@title Some mapping code (using the Folium library)
import folium

def get_polyline(lap, colour):
    return folium.PolyLine(locations=lap.tolist(), color=colour)

def plot_maps(laps):
    m = folium.Map(
        width='100%', 
        location=[45.33, -121.69],
        zoom_start=12,
        tiles='openstreetmap'
    )
    if len(laps)>0:
        lap1 = laps[0]
        lap = lap1.add_to(m)
        for lap_ in laps[1:]:
            lap_.add_to(m)
        m.fit_bounds(lap.get_bounds())

    return m
```


```python
lap1 = get_polyline(get_lap(workout_df, laps, 1), 'red')
lap2 = get_polyline(get_lap(workout_df, laps, 2), 'green')
lap3 = get_polyline(get_lap(workout_df, laps, 3), 'blue')
lap6 = get_polyline(get_lap(workout_df, laps, 6), 'orange')
lap9 = get_polyline(get_lap(workout_df, laps, 9), 'brown')
```


```python
#GPS data is messy as hell, just this plot to illustrate it
plot_maps([lap6, lap9])
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVM9ZmFsc2U7IExfTk9fVE9VQ0g9ZmFsc2U7IExfRElTQUJMRV8zRD1mYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS40LjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NvZGUuanF1ZXJ5LmNvbS9qcXVlcnktMS4xMi40Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS40LjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdjZG4uZ2l0aGFjay5jb20vcHl0aG9uLXZpc3VhbGl6YXRpb24vZm9saXVtL21hc3Rlci9mb2xpdW0vdGVtcGxhdGVzL2xlYWZsZXQuYXdlc29tZS5yb3RhdGUuY3NzIi8+CiAgICA8c3R5bGU+aHRtbCwgYm9keSB7d2lkdGg6IDEwMCU7aGVpZ2h0OiAxMDAlO21hcmdpbjogMDtwYWRkaW5nOiAwO308L3N0eWxlPgogICAgPHN0eWxlPiNtYXAge3Bvc2l0aW9uOmFic29sdXRlO3RvcDowO2JvdHRvbTowO3JpZ2h0OjA7bGVmdDowO308L3N0eWxlPgogICAgCiAgICA8bWV0YSBuYW1lPSJ2aWV3cG9ydCIgY29udGVudD0id2lkdGg9ZGV2aWNlLXdpZHRoLAogICAgICAgIGluaXRpYWwtc2NhbGU9MS4wLCBtYXhpbXVtLXNjYWxlPTEuMCwgdXNlci1zY2FsYWJsZT1ubyIgLz4KICAgIDxzdHlsZT4jbWFwXzgyNDEyYWY1NGI1NzRlMTNiYmU0MzIwMWRmMWYxNzM1IHsKICAgICAgICBwb3NpdGlvbjogcmVsYXRpdmU7CiAgICAgICAgd2lkdGg6IDEwMC4wJTsKICAgICAgICBoZWlnaHQ6IDEwMC4wJTsKICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgIHRvcDogMC4wJTsKICAgICAgICB9CiAgICA8L3N0eWxlPgo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgPGRpdiBjbGFzcz0iZm9saXVtLW1hcCIgaWQ9Im1hcF84MjQxMmFmNTRiNTc0ZTEzYmJlNDMyMDFkZjFmMTczNSIgPjwvZGl2Pgo8L2JvZHk+CjxzY3JpcHQ+ICAgIAogICAgCiAgICAKICAgICAgICB2YXIgYm91bmRzID0gbnVsbDsKICAgIAoKICAgIHZhciBtYXBfODI0MTJhZjU0YjU3NGUxM2JiZTQzMjAxZGYxZjE3MzUgPSBMLm1hcCgKICAgICAgICAnbWFwXzgyNDEyYWY1NGI1NzRlMTNiYmU0MzIwMWRmMWYxNzM1JywgewogICAgICAgIGNlbnRlcjogWzQ1LjMzLCAtMTIxLjY5XSwKICAgICAgICB6b29tOiAxMiwKICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICBsYXllcnM6IFtdLAogICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcsCiAgICAgICAgem9vbUNvbnRyb2w6IHRydWUsCiAgICAgICAgfSk7CgoKICAgIAogICAgdmFyIHRpbGVfbGF5ZXJfM2M3Yzg2NjFhYjNiNDdlYmExYjE0N2I1NDMzNmJiYjYgPSBMLnRpbGVMYXllcigKICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgIHsKICAgICAgICAiYXR0cmlidXRpb24iOiBudWxsLAogICAgICAgICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAgICAgICAibWF4TmF0aXZlWm9vbSI6IDE4LAogICAgICAgICJtYXhab29tIjogMTgsCiAgICAgICAgIm1pblpvb20iOiAwLAogICAgICAgICJub1dyYXAiOiBmYWxzZSwKICAgICAgICAib3BhY2l0eSI6IDEsCiAgICAgICAgInN1YmRvbWFpbnMiOiAiYWJjIiwKICAgICAgICAidG1zIjogZmFsc2UKfSkuYWRkVG8obWFwXzgyNDEyYWY1NGI1NzRlMTNiYmU0MzIwMWRmMWYxNzM1KTsKICAgIAogICAgICAgICAgICAgICAgdmFyIHBvbHlfbGluZV8yM2QwOTZhY2ZkZmI0YzQ0ODY5ZmNlMmM0NDU2ZDk1MiA9IEwucG9seWxpbmUoCiAgICAgICAgICAgICAgICAgICAgW1s1MS4yMzc3OTEyMTc4NjM1NiwgMy4zOTQ5MDg0MTE0MjgzMzIzXSwgWzUxLjIzNzczNDU1NjE5ODEyLCAzLjM5NDk1MTc0NTg2NzcyOV0sIFs1MS4yMzc2NzYyMTgxNTIwNDYsIDMuMzk0OTkxNzI3NTQ1ODU3NF0sIFs1MS4yMzc2MDQ1NTI4Nzk5MywgMy4zOTUwMTE2NzY0NzU0MDU3XSwgWzUxLjIzNzUzNjI0MDM2OTA4LCAzLjM5NTAyODM1NjQ2MjcxN10sIFs1MS4yMzc0NjI4OTg3MTYzMywgMy4zOTUwMzg0MTQ3NDY1MjNdLCBbNTEuMjM3Mzg2MjA0MzAyMzEsIDMuMzk1MDUwMDY1NTkxOTMxM10sIFs1MS4yMzczMDQ1NjQ1NjU0MiwgMy4zOTUwNjAwNDAwNTY3MDU1XSwgWzUxLjIzNzIyNDUxNzM5MDEzLCAzLjM5NTA3MDAxNDUyMTQ3OTZdLCBbNTEuMjM3MTQxMjAxMjcyNjEsIDMuMzk1MDc4Mzk2NDI0NjUxXSwgWzUxLjIzNzA1OTU2MTUzNTcxNiwgMy4zOTUwODMzNDE3NDc1MjI0XSwgWzUxLjIzNjk3NjE2MTU5OTE2LCAzLjM5NTA5MDA0NzI3MDA1OTZdLCBbNTEuMjM2OTA0NDk2MzI3MDQsIDMuMzk1MDg4MzcwODg5NDI1M10sIFs1MS4yMzY4MzI4MzEwNTQ5MjYsIDMuMzk1MDgzMzQxNzQ3NTIyNF0sIFs1MS4yMzY3Nzc4NDU3NzAxMiwgMy4zOTUwNjY3NDU1NzkyNDI3XSwgWzUxLjIzNjcyNzg4OTYyNzIyLCAzLjM5NTA3MzM2NzI4Mjc0ODJdLCBbNTEuMjM2Njk2MjA2MDMzMjMsIDMuMzk1MTY2NzQxNjg0MDc5XSwgWzUxLjIzNjY4Mjg3ODgwNzE5LCAzLjM5NTIxNjY5NzgyNjk4MTVdLCBbNTEuMjM2Njg3ODI0MTMwMDYsIDMuMzk1MjU1MDAzMTI0NDc1NV0sIFs1MS4yMzY3MDYxODA0OTgwMDQsIDMuMzk1Mjc2NzEyMjUzNjg5OF0sIFs1MS4yMzY3NjExNjU3ODI4MSwgMy4zOTUyNzAwMDY3MzExNTI1XSwgWzUxLjIzNjgwOTUyOTM2NDExLCAzLjM5NTI1ODM1NTg4NTc0NF0sIFs1MS4yMzY4Njk1NDM3OTA4MiwgMy4zOTUyNDgzODE0MjA5N10sIFs1MS4yMzY5Mjc4ODE4MzY4OSwgMy4zOTUyMjY2NzIyOTE3NTU3XSwgWzUxLjIzNjk4NjIxOTg4Mjk2NSwgMy4zOTUyMTAwNzYxMjM0NzZdLCBbNTEuMjM3MDU2MjA4Nzc0NDUsIDMuMzk1MTkwMDQzMzc0ODk2XSwgWzUxLjIzNzEyMjg0NDkwNDY2LCAzLjM5NTE3MzM2MzM4NzU4NDddLCBbNTEuMjM3MTg3ODg4NDczMjcsIDMuMzk1MTU1MDA3MDE5NjM5XSwgWzUxLjIzNzI1Nzg3NzM2NDc1NSwgMy4zOTUxMzY3MzQ0NzA3MjVdLCBbNTEuMjM3MzI3ODY2MjU2MjQsIDMuMzk1MTEzMzQ4OTYwODc2NV0sIFs1MS4yMzczOTYxNzg3NjcwODUsIDMuMzk1MTAwMDIxNzM0ODMzN10sIFs1MS4yMzc0NjI4OTg3MTYzMywgMy4zOTUwODMzNDE3NDc1MjI0XSwgWzUxLjIzNzUzMjg4NzYwNzgxLCAzLjM5NTA2MDA0MDA1NjcwNTVdLCBbNTEuMjM3NjA0NTUyODc5OTMsIDMuMzk1MDM2NzM4MzY1ODg4Nl0sIFs1MS4yMzc2Nzc4OTQ1MzI2OCwgMy4zOTUwMTgzODE5OTc5NDNdLCBbNTEuMjM3NzUxMjM2MTg1NDMsIDMuMzk0OTk4MzQ5MjQ5MzYzXSwgWzUxLjIzNzgyNDQ5NDAxOTE1LCAzLjM5NDk2NTA3MzA5Mzc3Ml0sIFs1MS4yMzc4OTQ1NjY3Mjk2NjUsIDMuMzk0OTMzMzg5NDk5NzgzNV0sIFs1MS4yMzc5Njk1MDA5NDQwMiwgMy4zOTQ5MDE3MDU5MDU3OTVdLCBbNTEuMjM4MDQxMTY2MjE2MTM1LCAzLjM5NDg3NTA1MTQ1MzcwOTZdLCBbNTEuMjM4MTA5NTYyNTQ2MDE1LCAzLjM5NDg1MTY2NTk0Mzg2MV0sIFs1MS4yMzgxNzI4NDU5MTQ5NiwgMy4zOTQ4MjMzMzUxMTExNDFdLCBbNTEuMjM4MjIxMjA5NDk2MjYsIDMuMzk0Nzc4NDA4MTEwMTQxOF0sIFs1MS4yMzgyNDk1NDAzMjg5OCwgMy4zOTQ3MTMzNjQ1NDE1MzA2XSwgWzUxLjIzODI1MTIxNjcwOTYxNCwgMy4zOTQ2MzUwNzc1NjU5MDg0XSwgWzUxLjIzODI0OTU0MDMyODk4LCAzLjM5NDUzNTA4MTQ2MTA3Ml0sIFs1MS4yMzgyMzQ1MzY3MjIzLCAzLjM5NDQ0MTcwNzA1OTc0MV0sIFs1MS4yMzgyMzI4NjAzNDE2NywgMy4zOTQzNTY3MTQ1NjE1ODE2XSwgWzUxLjIzODI0OTU0MDMyODk4LCAzLjM5NDI2ODM2OTMwMjE1MzZdLCBbNTEuMjM4Mjg0NDkyODY1MjA1LCAzLjM5NDIxNjczNjc3ODYxN10sIFs1MS4yMzgzMjc4MjczMDQ2LCAzLjM5NDE5ODM4MDQxMDY3MTJdLCBbNTEuMjM4MzcxMTYxNzQ0LCAzLjM5NDIxMzM4NDAxNzM0ODNdLCBbNTEuMjM4NDA2MTk4MDk5MjU2LCAzLjM5NDI2ODM2OTMwMjE1MzZdLCBbNTEuMjM4NDMxMTc2MTcwNzEsIDMuMzk0MzU1MDM4MTgwOTQ3M10sIFs1MS4yMzg0NDYxNzk3NzczODQsIDMuMzk0NDU2NzEwNjY2NDE4XSwgWzUxLjIzODQ1OTUwNzAwMzQzLCAzLjM5NDU3MDAzMzk5NzI5NzNdLCBbNTEuMjM4NDcyODM0MjI5NDcsIDMuMzk0NjgzMzU3MzI4MTc2NV0sIFs1MS4yMzg0Nzk1Mzk3NTIwMSwgMy4zOTQ3ODE2NzcwNTIzNzg3XSwgWzUxLjIzODQ5MTE5MDU5NzQxNSwgMy4zOTQ4NzAwMjIzMTE4MDY3XSwgWzUxLjIzODUwMTE2NTA2MjE5LCAzLjM5NDk0MDAxMTIwMzI4OV0sIFs1MS4yMzg1MTI4OTk3MjY2MywgMy4zOTQ5NTMzMzg0MjkzMzJdLCBbNTEuMjM4NTE3ODQ1MDQ5NSwgMy4zOTQ5MzAwMzY3Mzg1MTVdLCBbNTEuMjM4NTIyODc0MTkxNCwgMy4zOTQ4ODY3MDIyOTkxMThdLCBbNTEuMjM4NTEyODk5NzI2NjMsIDMuMzk0Nzk4MzU3MDM5NjldLCBbNTEuMjM4NTAyODQxNDQyODIsIDMuMzk0NzE1MDQwOTIyMTY1XSwgWzUxLjIzODQ4OTUxNDIxNjc4LCAzLjM5NDYyMDA3Mzk1OTIzMTRdLCBbNTEuMjM4NDc5NTM5NzUyMDEsIDMuMzk0NTE2NzI1MDkzMTI2M10sIFs1MS4yMzg0NzExNTc4NDg4MzUsIDMuMzk0NDExNjk5ODQ2Mzg3XSwgWzUxLjIzODQ1NDU2MTY4MDU1NSwgMy4zOTQzMDgzNTA5ODAyODJdLCBbNTEuMjM4NDQ2MTc5Nzc3Mzg0LCAzLjM5NDIxNTA2MDM5Nzk4MjZdLCBbNTEuMjM4NDUxMjA4OTE5MjksIDMuMzk0MTE4MzMzMjM1MzgzXSwgWzUxLjIzODQ2NDUzNjE0NTMzLCAzLjM5NDA1ODQwMjYyNzcwNjVdLCBbNTEuMjM4NDg0NTY4ODkzOTEsIDMuMzk0MDQxNzIyNjQwMzk1XSwgWzUxLjIzODQ5NDU0MzM1ODY4NCwgMy4zOTQwNjgzNzcwOTI0ODA3XSwgWzUxLjIzODUwNDUxNzgyMzQ2LCAzLjM5NDEzMzMzNjg0MjA2XSwgWzUxLjIzODUxMTIyMzM0NTk5NSwgMy4zOTQyMjE2ODIxMDE0ODhdLCBbNTEuMjM4NTE5NTIxNDMwMTM1LCAzLjM5NDMyNTAzMDk2NzU5M10sIFs1MS4yMzg1Mzc4Nzc3OTgwOCwgMy4zOTQ0MzUwMDE1MzcyMDRdLCBbNTEuMjM4NTQ5NTI4NjQzNDksIDMuMzk0NTQwMDI2NzgzOTQzXSwgWzUxLjIzODU1OTUwMzEwODI2LCAzLjM5NDY0MTY5OTI2OTQxNF0sIFs1MS4yMzg1NjQ1MzIyNTAxNjYsIDMuMzk0NzMwMDQ0NTI4ODQyXSwgWzUxLjIzODU3MTIzNzc3MjcsIDMuMzk0ODAxNzA5ODAwOTU4Nl0sIFs1MS4yMzg1Nzc4NTk0NzYyMSwgMy4zOTQ4NjUwNzY5ODg5MzU1XSwgWzUxLjIzODU5Mjg2MzA4Mjg4NiwgMy4zOTQ5MDY3MzUwNDc2OThdLCBbNTEuMjM4NjAyODM3NTQ3NjYsIDMuMzk0OTAwMDI5NTI1MTYxXSwgWzUxLjIzODYwOTU0MzA3MDIsIDMuMzk0ODY2NjY5NTUwNTM4XSwgWzUxLjIzODYwMTE2MTE2NzAyNiwgMy4zOTQ4MTAwMDc4ODUwOTg1XSwgWzUxLjIzODU4NzgzMzk0MDk4LCAzLjM5NDc0MzM3MTc1NDg4NDddLCBbNTEuMjM4NTgyODg4NjE4MTEsIDMuMzk0NjYwMDU1NjM3MzU5Nl0sIFs1MS4yMzg1ODk1MTAzMjE2MiwgMy4zOTQ1NjgzNTc2MTY2NjNdLCBbNTEuMjM4NjMxMTY4MzgwMzgsIDMuMzk0NDkwMDcwNjQxMDQxXSwgWzUxLjIzODY3OTUzMTk2MTY4LCAzLjM5NDQyNjcwMzQ1MzA2NF0sIFs1MS4yMzg3MjYyMTkxNjIzNDUsIDMuMzk0MzUxNjg1NDE5Njc4N10sIFs1MS4yMzg3NjI4NDgwNzkyMDUsIDMuMzk0MjY4MzY5MzAyMTUzNl0sIFs1MS4yMzg3OTc4ODQ0MzQ0NiwgMy4zOTQxNzM0MDIzMzkyMl0sIFs1MS4yMzg4Mjc4OTE2NDc4MTYsIDMuMzk0MDc2Njc1MTc2NjIwNV0sIFs1MS4yMzg4NDk1MTY5NTgsIDMuMzkzOTg2NzM3MzU1NTldLCBbNTEuMjM4ODc2MTcxNDEwMDg0LCAzLjM5MzkwNTAxMzc5OTY2NzRdLCBbNTEuMjM4OTA3ODU1MDA0MDcsIDMuMzkzODM2NzAxMjg4ODE5M10sIFs1MS4yMzg5NDQ1Njc3Mzk5NiwgMy4zOTM3ODMzOTIzODQ2NDgzXSwgWzUxLjIzODk4NjIyNTc5ODcyNiwgMy4zOTM3NTAwMzI0MTAwMjU2XSwgWzUxLjIzOTAzMjgyOTE4MDM2LCAzLjM5Mzc0MzQxMDcwNjUyXSwgWzUxLjIzOTA2NDUxMjc3NDM1LCAzLjM5Mzc1MzM4NTE3MTI5NF0sIFs1MS4yMzkwNTc4OTEwNzA4NCwgMy4zOTM4MDAwNzIzNzE5NTk3XSwgWzUxLjIzOTAyNDUzMTA5NjIyLCAzLjM5Mzg4MTcxMjEwODg1MDVdLCBbNTEuMjM4OTUyODY1ODI0MSwgMy4zOTQwMDE3NDA5NjIyNjddLCBbNTEuMjM4ODg0NTUzMzEzMjU1LCAzLjM5NDEyODM5MTUxOTE4OV0sIFs1MS4yMzg4MzYxODk3MzE5NTYsIDMuMzk0MjM2Njg1NzA4MTY1XSwgWzUxLjIzODc5Mjg1NTI5MjU2LCAzLjM5NDM2MzMzNjI2NTA4N10sIFs1MS4yMzg3NTYyMjYzNzU3LCAzLjM5NDQ3MTcxNDI3MzA5NV0sIFs1MS4yMzg3MjExOTAwMjA0NCwgMy4zOTQ1NjgzNTc2MTY2NjNdLCBbNTEuMjM4Njk3ODg4MzI5NjI1LCAzLjM5NDY2MTczMjAxNzk5NF0sIFs1MS4yMzg2ODk1MDY0MjY0NTQsIDMuMzk0NzYwMDUxNzQyMTk2XSwgWzUxLjIzODcwMjgzMzY1MjQ5NiwgMy4zOTQ4NjM0MDA2MDgzMDFdLCBbNTEuMjM4NzM3ODcwMDA3NzUsIDMuMzk0OTUzMzM4NDI5MzMyXSwgWzUxLjIzODc5NDUzMTY3MzE5LCAzLjM5NDk5ODM0OTI0OTM2M10sIFs1MS4yMzg4NTI4Njk3MTkyNywgMy4zOTUwMzAwMzI4NDMzNTE0XSwgWzUxLjIzODkxMTIwNzc2NTM0LCAzLjM5NTA0MzM2MDA2OTM5NF0sIFs1MS4yMzg5NjYxOTMwNTAxNDYsIDMuMzk1MDI2NjgwMDgyMDgyN10sIFs1MS4yMzg5OTYyMDAyNjM1LCAzLjM5NTAxMDAwMDA5NDc3MTRdLCBbNTEuMjM4OTc3ODQzODk1NTU1LCAzLjM5NDk1NTAxNDgwOTk2Nl0sIFs1MS4yMzg5NTExODk0NDM0NywgMy4zOTQ5MTE2ODAzNzA1NjkyXSwgWzUxLjIzODkxNzgyOTQ2ODg0NiwgMy4zOTQ4NzUwNTE0NTM3MDk2XSwgWzUxLjIzODg4MTIwMDU1MTk5LCAzLjM5NDgyODM2NDI1MzA0NF0sIFs1MS4yMzg4NTk0OTE0MjI3NywgMy4zOTQ3NjE3MjgxMjI4MzA0XSwgWzUxLjIzODg2MTE2NzgwMzQxLCAzLjM5NDY4NTAzMzcwODgxMV0sIFs1MS4yMzg4OTI4NTEzOTczOTUsIDMuMzk0NjM2NjcwMTI3NTExXSwgWzUxLjIzODkyNzg4Nzc1MjY1LCAzLjM5NDYyODM3MjA0MzM3MV0sIFs1MS4yMzg5NjExNjM5MDgyNCwgMy4zOTQ2MTUwNDQ4MTczMjg1XSwgWzUxLjIzODk4NjIyNTc5ODcyNiwgMy4zOTQ1NzUwNjMxMzkyXSwgWzUxLjIzODk5NjIwMDI2MzUsIDMuMzk0NTA4MzQzMTg5OTU0OF0sIFs1MS4yMzg5OTYyMDAyNjM1LCAzLjM5NDQzNjY3NzkxNzgzOF0sIFs1MS4yMzg5NTExODk0NDM0NywgMy4zOTQzOTAwNzQ1MzYyMDQzXSwgWzUxLjIzODkxOTUwNTg0OTQ4LCAzLjM5NDMyNTAzMDk2NzU5M10sIFs1MS4yMzg5MDEyMzMzMDA1NywgMy4zOTQyNDUwNjc2MTEzMzY3XSwgWzUxLjIzODg5OTU1NjkxOTkzLCAzLjM5NDE2NTAyMDQzNjA0ODVdLCBbNTEuMjM4OTIxMTgyMjMwMTE1LCAzLjM5NDEwMTczNzA2NzEwMzRdLCBbNTEuMjM4OTY3ODY5NDMwNzgsIDMuMzk0MDg2NzMzNDYwNDI2M10sIFs1MS4yMzkwMTI4ODAyNTA4MSwgMy4zOTQxMjMzNjIzNzcyODZdLCBbNTEuMjM5MDQ0NTYzODQ0OCwgMy4zOTQxODAwMjQwNDI3MjU2XSwgWzUxLjIzOTA1Mjg2MTkyODk0LCAzLjM5NDI2MTc0NzU5ODY0OF0sIFs1MS4yMzkwNTQ1MzgzMDk1NzQsIDMuMzk0MzYxNzQzNzAzNDg0NV0sIFs1MS4yMzkwNzI4OTQ2Nzc1MiwgMy4zOTQ0ODUwNDE0OTkxMzhdLCBbNTEuMjM5MDg2MjIxOTAzNTYsIDMuMzk0NjA1MDcwMzUyNTU0M10sIFs1MS4yMzkwOTExNjcyMjY0MzQsIDMuMzk0NzIwMDcwMDY0MDY4XSwgWzUxLjIzOTA5MTE2NzIyNjQzNCwgMy4zOTQ4MjUwMTE0OTE3NzU1XSwgWzUxLjIzOTA5Mjg0MzYwNzA3LCAzLjM5NDkyMDA2MjI3Mzc0MDhdLCBbNTEuMjM5MDk3ODcyNzQ4OTcsIDMuMzk0OTk2NjcyODY4NzI4Nl0sIFs1MS4yMzkxMTk0OTgwNTkxNTQsIDMuMzk1MDIwMDU4Mzc4NTc3Ml0sIFs1MS4yMzkxMjc4Nzk5NjIzMjUsIDMuMzk0OTk4MzQ5MjQ5MzYzXSwgWzUxLjIzOTEyOTU1NjM0Mjk2LCAzLjM5NDk1MDA2OTQ4NzA5NV0sIFs1MS4yMzkxMjYyMDM1ODE2OSwgMy4zOTQ4OTAwNTUwNjAzODY3XSwgWzUxLjIzOTExNjIyOTExNjkyLCAzLjM5NDgxMTY4NDI2NTczMjhdLCBbNTEuMjM5MTA2MTcwODMzMTEsIDMuMzk0NzI1MDE1Mzg2OTM5XSwgWzUxLjIzOTEwMTIyNTUxMDI0LCAzLjM5NDYyNjY5NTY2MjczN10sIFs1MS4yMzkwOTk1NDkxMjk2MDUsIDMuMzk0NTIwMDc3ODU0Mzk1XSwgWzUxLjIzOTA5OTU0OTEyOTYwNSwgMy4zOTQ0MTY3Mjg5ODgyOV0sIFs1MS4yMzkxMDQ0OTQ0NTI0OCwgMy4zOTQzMTY3MzI4ODM0NTM0XSwgWzUxLjIzOTEyMTE3NDQzOTc5LCAzLjM5NDIyNjcxMTI0MzM5MV0sIFs1MS4yMzkxNjI4MzI0OTg1NSwgMy4zOTQxNjAwNzUxMTMxNzczXSwgWzUxLjIzOTIxMjg3MjQ2MDQ4NSwgMy4zOTQxMDM0MTM0NDc3Mzc3XSwgWzUxLjIzOTI3NDU2MzI2NzgzLCAzLjM5NDEwNTAwNjAwOTM0MDNdLCBbNTEuMjM5MzIxMTY2NjQ5NDYsIDMuMzk0MTM4MzY1OTgzOTYzXSwgWzUxLjIzOTM1NjIwMzAwNDcyLCAzLjM5NDIwNTAwMjExNDE3NjhdLCBbNTEuMjM5MzU2MjAzMDA0NzIsIDMuMzk0MjkxNjcwOTkyOTcwNV0sIFs1MS4yMzkzMzYxNzAyNTYxNCwgMy4zOTQzNzY3NDczMTAxNjE2XSwgWzUxLjIzOTMwOTUxNTgwNDA1LCAzLjM5NDQ1ODM4NzA0NzA1MjRdLCBbNTEuMjM5Mjg3ODkwNDkzODcsIDMuMzk0NTQ2NzMyMzA2NDgwNF0sIFs1MS4yMzkyNzc4MzIyMTAwNjQsIDMuMzk0NjQ1MDUyMDMwNjgyNl0sIFs1MS4yMzkyODExODQ5NzEzMywgMy4zOTQ3NDAwMTg5OTM2MTZdLCBbNTEuMjM5Mjg3ODkwNDkzODcsIDMuMzk0ODI4MzY0MjUzMDQ0XSwgWzUxLjIzOTI5MTE1OTQzNjExLCAzLjM5NDkwMzM4MjI4NjQyOTRdLCBbNTEuMjM5MzA0NTcwNDgxMTgsIDMuMzk0OTQ1MDQwMzQ1MTkyXSwgWzUxLjIzOTMxNjIyMTMyNjU5LCAzLjM5NDkzMDAzNjczODUxNV0sIFs1MS4yMzkzMjI4NDMwMzAwOTUsIDMuMzk0ODkxNzMxNDQxMDIxXSwgWzUxLjIzOTMwNzgzOTQyMzQyLCAzLjM5NDgzMDA0MDYzMzY3ODRdLCBbNTEuMjM5Mjg3ODkwNDkzODcsIDMuMzk0NzU1MDIyNjAwMjkzXSwgWzUxLjIzOTI3OTUwODU5MDcsIDMuMzk0NjY2Njc3MzQwODY1XSwgWzUxLjIzOTI3OTUwODU5MDcsIDMuMzk0NTc1MDYzMTM5Ml0sIFs1MS4yMzkyOTk1NDEzMzkyOCwgMy4zOTQ0NzUwNjcwMzQzNjM3XSwgWzUxLjIzOTMzMjkwMTMxMzksIDMuMzk0Mzg1MDQ1Mzk0MzAxNF0sIFs1MS4yMzkzOTQ1MDgzMDIyMSwgMy4zOTQzNzMzOTQ1NDg4OTNdLCBbNTEuMjM5NDU0NTIyNzI4OTIsIDMuMzk0NDE2NzI4OTg4MjldLCBbNTEuMjM5NDk0NTA0NDA3MDUsIDMuMzk0NDY2Njg1MTMxMTkyXSwgWzUxLjIzOTUyMTE1ODg1OTEzNCwgMy4zOTQ1MjgzNzU5Mzg1MzQ3XSwgWzUxLjIzOTUwMjg4NjMxMDIyLCAzLjM5NDYxMzM2ODQzNjY5NF0sIFs1MS4yMzk0ODI4NTM1NjE2NCwgMy4zOTQ3MDY3NDI4MzgwMjVdLCBbNTEuMjM5NDcxMjAyNzE2MjMsIDMuMzk0ODA2NzM4OTQyODYxNl0sIFs1MS4yMzk0Njk1MjYzMzU2LCAzLjM5NDkwNjczNTA0NzY5OF0sIFs1MS4yMzk0ODI4NTM1NjE2NCwgMy4zOTQ5OTE3Mjc1NDU4NTc0XSwgWzUxLjIzOTUwMjg4NjMxMDIyLCAzLjM5NTA2Njc0NTU3OTI0MjddLCBbNTEuMjM5NTI3ODY0MzgxNjcsIDMuMzk1MTEzMzQ4OTYwODc2NV0sIFs1MS4yMzk1NTI4NDI0NTMxMiwgMy4zOTUwODY2OTQ1MDg3OTFdLCBbNTEuMjM5NTg5NTU1MTg5MDEsIDMuMzk1MDA1MDU0NzcxOV0sIFs1MS4yMzk1OTQ1MDA1MTE4ODUsIDMuMzk0OTIzNDE1MDM1MDA5NF0sIFs1MS4yMzk2MDEyMDYwMzQ0MiwgMy4zOTQ4MzY3NDYxNTYyMTU3XSwgWzUxLjIzOTYwNzgyNzczNzkzLCAzLjM5NDc0NjcyNDUxNjE1MzNdLCBbNTEuMjM5NjIxMjM4NzgzLCAzLjM5NDY1ODM3OTI1NjcyNTNdLCBbNTEuMjM5NjM2MTU4NTcwNjUsIDMuMzk0NTgzMzYxMjIzMzRdLCBbNTEuMjM5NjQ3ODkzMjM1MDksIDMuMzk0NTA1MDc0MjQ3NzE4XSwgWzUxLjIzOTY1MTE2MjE3NzMyNCwgMy4zOTQ0MjUwMjcwNzI0Mjk3XSwgWzUxLjIzOTY1OTU0NDA4MDQ5NiwgMy4zOTQzMzE3MzY0OTAxMzA0XSwgWzUxLjIzOTY2NjE2NTc4NCwgMy4zOTQyNDAwMzg0Njk0MzRdLCBbNTEuMjM5NjY0NTczMjIyNCwgMy4zOTQxNjUwMjA0MzYwNDg1XSwgWzUxLjIzOTY2NDU3MzIyMjQsIDMuMzk0MDkwMDAyNDAyNjYzMl0sIFs1MS4yMzk2NjI4OTY4NDE3NjQsIDMuMzkzOTkzMzU5MDU5MDk1NF0sIFs1MS4yMzk2NTc4Njc2OTk4NiwgMy4zOTM4ODY3NDEyNTA3NTM0XSwgWzUxLjIzOTY0Mjg2NDA5MzE4NCwgMy4zOTM3ODgzMzc3MDc1MTk1XSwgWzUxLjIzOTYzMjg4OTYyODQxLCAzLjM5MzY4ODM0MTYwMjY4M10sIFs1MS4yMzk2MzYxNTg1NzA2NSwgMy4zOTM1OTUwNTEwMjAzODRdLCBbNTEuMjM5NjYxMjIwNDYxMTMsIDMuMzkzNTIzMzg1NzQ4MjY3XSwgWzUxLjIzOTcwMjg3ODUxOTg5LCAzLjM5MzQ4MDA1MTMwODg3MDNdLCBbNTEuMjM5NzUyODM0NjYyNzk1LCAzLjM5MzQ1NTA3MzIzNzQxOV0sIFs1MS4yMzk4MTExNzI3MDg4NywgMy4zOTM0MzgzOTMyNTAxMDc4XSwgWzUxLjIzOTg3Mjg2MzUxNjIxLCAzLjM5MzQyMTcxMzI2Mjc5NjRdLCBbNTEuMjM5OTQyODUyNDA3Njk0LCAzLjM5MzQwMzM1Njg5NDg1MDddLCBbNTEuMjQwMDEyODQxMjk5MTc2LCAzLjM5MzM4NTAwMDUyNjkwNV0sIFs1MS4yNDAwODk1MzU3MTMxOTYsIDMuMzkzMzcwMDgwNzM5MjU5N10sIFs1MS4yNDAxNjI4NzczNjU5NSwgMy4zOTMzNTgzNDYwNzQ4MTk2XSwgWzUxLjI0MDIzMTE4OTg3Njc5NSwgMy4zOTMzNTUwNzcxMzI1ODI3XSwgWzUxLjI0MDI5OTUwMjM4NzY0LCAzLjM5MzM1NTA3NzEzMjU4MjddLCBbNTEuMjQwMzcxMTY3NjU5NzYsIDMuMzkzMzU1MDc3MTMyNTgyN10sIFs1MS4yNDA0NDI4MzI5MzE4NzYsIDMuMzkzMzUxNzI0MzcxMzE0XSwgWzUxLjI0MDUxMTIyOTI2MTc1NiwgMy4zOTMzNDUwMTg4NDg3NzddLCBbNTEuMjQwNTc0NTEyNjMwNywgMy4zOTMzMzY3MjA3NjQ2MzddLCBbNTEuMjQwNjI0NTUyNTkyNjM1LCAzLjM5MzMxMzMzNTI1NDc4ODRdLCBbNTEuMjQwNjcyODMyMzU0OSwgMy4zOTMyODE3MzU0Nzk4MzE3XSwgWzUxLjI0MDY3NDUwODczNTU0LCAzLjM5MzE5ODMzNTU0MzI3NV0sIFs1MS4yNDA2Njk1NjM0MTI2NjYsIDMuMzkzMTI2NjcwMjcxMTU4XSwgWzUxLjI0MDYzMjg1MDY3Njc3NSwgMy4zOTMxMTAwNzQxMDI4Nzg2XSwgWzUxLjI0MDU4OTUxNjIzNzM4LCAzLjM5MzEwMTY5MjE5OTcwN10sIFs1MS4yNDA1NDEyMzY0NzUxMSwgMy4zOTMwOTY3NDY4NzY4MzZdLCBbNTEuMjQwNDcyODQwMTQ1MjMsIDMuMzkzMDY1MDYzMjgyODQ3NF0sIFs1MS4yNDA0MTc4NTQ4NjA0MjUsIDMuMzkzMDE4Mzc2MDgyMTgyXSwgWzUxLjI0MDM5OTQ5ODQ5MjQ4LCAzLjM5Mjk0MzM1ODA0ODc5NjddLCBbNTEuMjQwMzkyODc2Nzg4OTc0LCAzLjM5Mjg3MTY5Mjc3NjY4XSwgWzUxLjI0MDQxOTUzMTI0MTA2LCAzLjM5MjgxMTY3ODM0OTk3MThdLCBbNTEuMjQwNDU3ODM2NTM4NTUsIDMuMzkyODAzMzgwMjY1ODMyXSwgWzUxLjI0MDQ4NDQ5MDk5MDY0LCAzLjM5MjgyMzQxMzAxNDQxMl0sIFs1MS4yNDA1MTQ0OTgyMDM5OSwgMy4zOTI4NTMzMzY0MDg3MzQzXSwgWzUxLjI0MDU0MTIzNjQ3NTExLCAzLjM5Mjg2MzM5NDY5MjU0XSwgWzUxLjI0MDU2NjIxNDU0NjU2LCAzLjM5Mjg0MzM2MTk0Mzk2XSwgWzUxLjI0MDU3NDUxMjYzMDcsIDMuMzkyNzkzNDA1ODAxMDU4XSwgWzUxLjI0MDU3MTE1OTg2OTQzLCAzLjM5MjczODMzNjY5NzIyMV0sIFs1MS4yNDA1NjI4NjE3ODUyOSwgMy4zOTI2ODUwMjc3OTMwNV0sIFs1MS4yNDA1NDc4NTgxNzg2MTYsIDMuMzkyNjMxNzE4ODg4ODc5XSwgWzUxLjI0MDUwNDUyMzczOTIyLCAzLjM5MjYzODM0MDU5MjM4NDNdLCBbNTEuMjQwNDY5NTcxMjAyOTksIDMuMzkyNjMxNzE4ODg4ODc5XSwgWzUxLjI0MDQyNDU2MDM4Mjk2LCAzLjM5MjYyMTc0NDQyNDEwNDddLCBbNTEuMjQwMzcyODQ0MDQwMzk0LCAzLjM5MjYyODM2NjEyNzYxXSwgWzUxLjI0MDMyMjg4Nzg5NzQ5LCAzLjM5MjYzMDA0MjUwODI0NDVdLCBbNTEuMjQwMjY3OTAyNjEyNjg2LCAzLjM5MjYzODM0MDU5MjM4NDNdLCBbNTEuMjQwMjA0NTM1NDI0NzEsIDMuMzkyNjU1MDIwNTc5Njk1N10sIFs1MS4yNDAxNDExNjgyMzY3MywgMy4zOTI2NzAwMjQxODYzNzI4XSwgWzUxLjI0MDA2OTUwMjk2NDYxNiwgMy4zOTI2OTAwNTY5MzQ5NTI3XSwgWzUxLjI0MDAwMTE5MDQ1Mzc3LCAzLjM5MjY5ODM1NTAxOTA5MjZdLCBbNTEuMjM5OTQ2MjA1MTY4OTYsIDMuMzkyNjk4MzU1MDE5MDkyNl0sIFs1MS4yMzk5MDI4NzA3Mjk1NjYsIDMuMzkyNzI2Njg1ODUxODEyNF0sIFs1MS4yMzk4ODYxOTA3NDIyNTQsIDMuMzkyNzgzMzQ3NTE3MjUyXSwgWzUxLjIzOTg5Mjg5NjI2NDc5LCAzLjM5MjgyMDA2MDI1MzE0MzNdLCBbNTEuMjM5OTE0NTIxNTc0OTc0LCAzLjM5Mjg0ODM5MTA4NTg2M10sIFs1MS4yMzk5Mzk0OTk2NDY0MjUsIDMuMzkyODYzMzk0NjkyNTRdLCBbNTEuMjM5OTc0NTM2MDAxNjgsIDMuMzkyODYwMDQxOTMxMjcxNl0sIFs1MS4yNDAwMjI4OTk1ODI5OCwgMy4zOTI4NTUwMTI3ODkzNjg2XSwgWzUxLjI0MDA3Mjg1NTcyNTg4NCwgMy4zOTI4NDMzNjE5NDM5Nl0sIFs1MS4yNDAxMjYxNjQ2MzAwNTUsIDMuMzkyODMxNzExMDk4NTUxOF0sIFs1MS4yNDAxOTc4Mjk5MDIxNywgMy4zOTI4MDg0MDk0MDc3MzVdLCBbNTEuMjQwMjYyODczNDcwNzgsIDMuMzkyNzkxNzI5NDIwNDIzNV0sIFs1MS4yNDAzMzExODU5ODE2MywgMy4zOTI3NjgzNDM5MTA1NzVdLCBbNTEuMjQwMzk0NTUzMTY5NjEsIDMuMzkyNzQxNjg5NDU4NDg5NF0sIFs1MS4yNDA0NjI4NjU2ODA0NTYsIDMuMzkyNzIxNzQwNTI4OTQxXSwgWzUxLjI0MDUyNjIzMjg2ODQzLCAzLjM5MjcyMzMzMzA5MDU0MzddLCBbNTEuMjQwNTc3ODY1MzkxOTcsIDMuMzkyNzQxNjg5NDU4NDg5NF0sIFs1MS4yNDA2MTk1MjM0NTA3MywgMy4zOTI3NzAwMjAyOTEyMDldLCBbNTEuMjQwNTc5NTQxNzcyNjA0LCAzLjM5MjgyODM1ODMzNzI4M10sIFs1MS4yNDA1NjYyMTQ1NDY1NiwgMy4zOTI4NTAwNjc0NjY0OTc0XSwgWzUxLjI0MDU0NDUwNTQxNzM1LCAzLjM5MjgxMzM1NDczMDYwNl0sIFs1MS4yNDA1MTk1MjczNDU4OTYsIDMuMzkyNzczMzczMDUyNDc4XSwgWzUxLjI0MDQ3NjE5MjkwNjUsIDMuMzkyNzU4MzY5NDQ1ODAxXSwgWzUxLjI0MDQzOTU2Mzk4OTY0LCAzLjM5Mjc2NTA3NDk2ODMzOF0sIFs1MS4yNDAzOTk0OTg0OTI0OCwgMy4zOTI3ODAwNzg1NzUwMTVdLCBbNTEuMjQwMzE5NTM1MTM2MjIsIDMuMzkyODIwMDYwMjUzMTQzM10sIFs1MS4yNDAyNTQ0OTE1Njc2MSwgMy4zOTI4NDMzNjE5NDM5Nl0sIFs1MS4yNDAxODYxNzkwNTY3NjQsIDMuMzkyODY4MzQwMDE1NDExNF0sIFs1MS4yNDAxMTYxOTAxNjUyOCwgMy4zOTI4OTE3MjU1MjUyNl0sIFs1MS4yNDAwNDc4Nzc2NTQ0MywgMy4zOTI5MDUwNTI3NTEzMDI3XSwgWzUxLjIzOTk4NDUxMDQ2NjQ1NiwgMy4zOTI5MTUwMjcyMTYwNzddLCBbNTEuMjM5OTI2MTcyNDIwMzgsIDMuMzkyOTE4Mzc5OTc3MzQ1NV0sIFs1MS4yMzk4NzYyMTYyNzc0OCwgMy4zOTI5MjUwMDE2ODA4NTFdLCBbNTEuMjM5ODM2MjM0NTk5MzUsIDMuMzkyOTQ2NzEwODEwMDY1M10sIFs1MS4yMzk4MzQ1NTgyMTg3MiwgMy4zOTI5ODAwNzA3ODQ2ODhdLCBbNTEuMjM5ODU3ODU5OTA5NTM0LCAzLjM5MzAwNTA0ODg1NjEzOV0sIFs1MS4yMzk4ODQ1MTQzNjE2MiwgMy4zOTMwMDAwMTk3MTQyMzYzXSwgWzUxLjIzOTkyOTUyNTE4MTY1LCAzLjM5Mjk5MTcyMTYzMDA5NjRdLCBbNTEuMjM5OTk5NTE0MDczMTMsIDMuMzkyOTk2NjY2OTUyOTY3Nl0sIFs1MS4yNDAwNjYyMzQwMjIzOCwgMy4zOTI5NzgzOTQ0MDQwNTM3XSwgWzUxLjI0MDEzNzg5OTI5NDQ5NiwgMy4zOTI5NjAwMzgwMzYxMDhdLCBbNTEuMjQwMjAxMTgyNjYzNDQsIDMuMzkyOTM1MDU5OTY0NjU3XSwgWzUxLjI0MDI1Nzg0NDMyODg4LCAzLjM5MjkxMzM1MDgzNTQ0MjVdLCBbNTEuMjQwMzA0NTMxNTI5NTQ2LCAzLjM5MjkyMDA1NjM1Nzk3OThdLCBbNTEuMjQwMzExMjM3MDUyMDgsIDMuMzkyOTUxNzM5OTUxOTY4XSwgWzUxLjI0MDMxMjgyOTYxMzY4NiwgMy4zOTI5ODgzNjg4Njg4MjhdLCBbNTEuMjQwMjcxMTcxNTU0OTIsIDMuMzkyOTk2NjY2OTUyOTY3Nl0sIFs1MS4yNDAyMTYxODYyNzAxMiwgMy4zOTMwMDY3MjUyMzY3NzM1XSwgWzUxLjI0MDE0Mjg0NDYxNzM3LCAzLjM5MzAyMDA1MjQ2MjgxNjJdLCBbNTEuMjQwMDc3ODg0ODY3NzksIDMuMzkzMDI1MDgxNjA0NzE5XSwgWzUxLjI0MDAxNDUxNzY3OTgxLCAzLjM5MzAzNjczMjQ1MDEyNzZdLCBbNTEuMjM5OTY3ODMwNDc5MTQ1LCAzLjM5MzA2MDAzNDE0MDk0NDVdLCBbNTEuMjM5OTQ3ODgxNTQ5NiwgMy4zOTMwOTAwNDEzNTQyOTg2XSwgWzUxLjIzOTkzNjIzMDcwNDE5LCAzLjM5MzEyNjY3MDI3MTE1OF0sIFs1MS4yMzk5NTQ1MDMyNTMxLCAzLjM5MzE0NTAyNjYzOTEwNF0sIFs1MS4yNDAwMTExNjQ5MTg1NCwgMy4zOTMxNTM0MDg1NDIyNzU0XSwgWzUxLjI0MDA2NjIzNDAyMjM4LCAzLjM5MzE0NTAyNjYzOTEwNF0sIFs1MS4yNDAxMjQ1NzIwNjg0NSwgMy4zOTMxMjAwNDg1Njc2NTI3XSwgWzUxLjI0MDE5MTIwODE5ODY3LCAzLjM5MzA5MzM5NDExNTU2N10sIFs1MS4yNDAyNTc4NDQzMjg4OCwgMy4zOTMwODAwNjY4ODk1MjQ1XSwgWzUxLjI0MDMyMTIxMTUxNjg2LCAzLjM5MzEwMDAxNTgxOTA3MjddLCBbNTEuMjQwMzgxMjI1OTQzNTY1LCAzLjM5MzEzMTY5OTQxMzA2MV0sIFs1MS4yNDA0Mzk1NjM5ODk2NCwgMy4zOTMxNjE3MDY2MjY0MTUzXSwgWzUxLjI0MDQ4OTUyMDEzMjU0LCAzLjM5MzE4ODM2MTA3ODUwMDddLCBbNTEuMjQwNTAyODQ3MzU4NTg0LCAzLjM5MzIzNjcyNDY1OTgwMDVdLCBbNTEuMjQwNDgyODk4NDI5MDM2LCAzLjM5MzI3MTY3NzE5NjAyNl0sIFs1MS4yNDA0NTEyMTQ4MzUwNSwgMy4zOTMyNzAwMDA4MTUzOTE1XSwgWzUxLjI0MDQxMTIzMzE1NjkyLCAzLjM5MzI0MzM0NjM2MzMwNl0sIFs1MS4yNDAzNTQ1NzE0OTE0OCwgMy4zOTMyMDAwMTE5MjM5MDldLCBbNTEuMjQwMjc5NTUzNDU4MDk1LCAzLjM5MzE0MDA4MTMxNjIzMjddLCBbNTEuMjQwMjE0NTA5ODg5NDgsIDMuMzkzMDk1MDcwNDk2MjAxNV0sIFs1MS4yNDAxNDc4NzM3NTkyNywgMy4zOTMwOTAwNDEzNTQyOTg2XSwgWzUxLjI0MDA4NDUwNjU3MTI5LCAzLjM5MzExMTY2NjY2NDQ4MV0sIFs1MS4yNDAwMjk1MjEyODY0OSwgMy4zOTMxNDgzNzk0MDAzNzI1XSwgWzUxLjIzOTk3MTE4MzI0MDQxNCwgMy4zOTMxNzMzNTc0NzE4MjM3XSwgWzUxLjIzOTkxMjg0NTE5NDM0LCAzLjM5MzE4MDA2Mjk5NDM2MV0sIFs1MS4yMzk4NTI4MzA3Njc2MywgMy4zOTMxNjUwNTkzODc2ODRdLCBbNTEuMjM5Nzk2MTY5MTAyMTksIDMuMzkzMTM1MDUyMTc0MzI5OF0sIFs1MS4yMzk3NTk1NDAxODUzMywgMy4zOTMxMjUwNzc3MDk1NTU2XSwgWzUxLjIzOTcxNDUyOTM2NTMsIDMuMzkzMTE4MzcyMTg3MDE4NF0sIFs1MS4yMzk2NTc4Njc2OTk4NiwgMy4zOTMxMTE2NjY2NjQ0ODFdLCBbNTEuMjM5NjAyODgyNDE1MDU2LCAzLjM5MzA5NTA3MDQ5NjIwMTVdLCBbNTEuMjM5NTUyODQyNDUzMTIsIDMuMzkzMDcxNjg0OTg2MzUzXSwgWzUxLjIzOTU0Mjg2Nzk4ODM1LCAzLjM5MzAxNjY5OTcwMTU0NzZdLCBbNTEuMjM5NTM5NTE1MjI3MDgsIDMuMzkyOTYxNzE0NDE2NzQyM10sIFs1MS4yMzk1ODExNzMyODU4NCwgMy4zOTI5NTUwMDg4OTQyMDVdLCBbNTEuMjM5NjIxMjM4NzgzLCAzLjM5Mjk0MzM1ODA0ODc5NjddLCBbNTEuMjM5NjY2MTY1Nzg0LCAzLjM5Mjk0NTAzNDQyOTQzMV0sIFs1MS4yMzk3MDc4MjM4NDI3NjQsIDMuMzkyOTgzMzM5NzI2OTI1XSwgWzUxLjIzOTczNzgzMTA1NjEyLCAzLjM5MzAzMzM3OTY4ODg1OV0sIFs1MS4yMzk3NjQ1NjkzMjcyMzUsIDMuMzkzMDk2NzQ2ODc2ODM2XSwgWzUxLjIzOTc4Nzg3MTAxODA1LCAzLjM5MzE1NTAwMTEwMzg3OF0sIFs1MS4yMzk4MDExOTgyNDQwOTUsIDMuMzkzMjE1MDE1NTMwNTg2Ml0sIFs1MS4yMzk3OTYxNjkxMDIxOSwgMy4zOTMyNjMzNzkxMTE4ODZdLCBbNTEuMjM5NzgxMTY1NDk1NTE1LCAzLjM5MzI5NjczOTA4NjUwODhdLCBbNTEuMjM5NzQ5NTY1NzIwNTYsIDMuMzkzMjk1MDYyNzA1ODc0NF0sIFs1MS4yMzk3MjYxODAyMTA3MSwgMy4zOTMyODM0MTE4NjA0NjZdLCBbNTEuMjM5NzA3ODIzODQyNzY0LCAzLjM5MzI1ODM0OTk2OTk4M10sIFs1MS4yMzk2OTI5MDQwNTUxMiwgMy4zOTMyMjMzOTc0MzM3NThdLCBbNTEuMjM5Njc5NDkzMDEwMDQ0LCAzLjM5MzIwNTA0MTA2NTgxMl0sIFs1MS4yMzk2NzI4NzEzMDY1NCwgMy4zOTMyMDY3MTc0NDY0NDY0XSwgWzUxLjIzOTY2NjE2NTc4NCwgMy4zOTMyNDgzNzU1MDUyMDldLCBbNTEuMjM5NjUyODM4NTU3OTYsIDMuMzkzMzE1MDExNjM1NDIyN10sIFs1MS4yMzk2MzEyMTMyNDc3NzYsIDMuMzkzMzgwMDU1MjA0MDM0XSwgWzUxLjIzOTU5OTUyOTY1Mzc5LCAzLjM5MzQ0NTAxNDk1MzYxMzNdLCBbNTEuMjM5NTUyODQyNDUzMTIsIDMuMzkzNTA2NzA1NzYwOTU2XSwgWzUxLjIzOTUwMjg4NjMxMDIyLCAzLjM5MzU1Njc0NTcyMjg5XSwgWzUxLjIzOTQ0Mjg3MTg4MzUxLCAzLjM5MzU5MDAyMTg3ODQ4MV0sIFs1MS4yMzkzNjYxNzc0Njk0OSwgMy4zOTM2MTM0MDczODgzMjk1XSwgWzUxLjIzOTI4NjIxNDExMzIzNSwgMy4zOTM2Mjg0MTA5OTUwMDY2XSwgWzUxLjIzOTIwNzg0MzMxODU4LCAzLjM5MzY0MDA2MTg0MDQxNV0sIFs1MS4yMzkxMzI4MjUyODUxOTYsIDMuMzkzNjU1MDY1NDQ3MDkyXSwgWzUxLjIzOTA1MTE4NTU0ODMwNiwgMy4zOTM2NzUwMTQzNzY2NDAzXSwgWzUxLjIzODk2Nzg2OTQzMDc4LCAzLjM5MzY5ODM5OTg4NjQ4OV0sIFs1MS4yMzg4OTExNzUwMTY3NiwgMy4zOTM3MzUwMjg4MDMzNDg1XSwgWzUxLjIzODgxMjg4ODA0MTE0LCAzLjM5Mzc2ODM4ODc3Nzk3MTNdLCBbNTEuMjM4NzI3ODk1NTQyOTgsIDMuMzkzNzkxNjkwNDY4Nzg4XSwgWzUxLjIzODY0NjE3MTk4NzA2LCAzLjM5MzgxMDA0NjgzNjczNF0sIFs1MS4yMzg1NjI4NTU4Njk1MywgMy4zOTM4MjY3MjY4MjQwNDVdLCBbNTEuMjM4NDgyODkyNTEzMjc1LCAzLjM5Mzg0NDk5OTM3Mjk1OV0sIFs1MS4yMzg0MDExNjg5NTczNSwgMy4zOTM4NjY3MDg1MDIxNzM0XSwgWzUxLjIzODMxNDUwMDA3ODU2LCAzLjM5Mzg4ODMzMzgxMjM1Nl0sIFs1MS4yMzgyMzI4NjAzNDE2NywgMy4zOTM5MDY2OTAxODAzMDE3XSwgWzUxLjIzODE0Nzg2Nzg0MzUxLCAzLjM5MzkyODM5OTMwOTUxNl0sIFs1MS4yMzgwNzI4NDk4MTAxMiwgMy4zOTM5NjE2NzU0NjUxMDddLCBbNTEuMjM4MDAxMTg0NTM4MDEsIDMuMzk0MDIxNjg5ODkxODE1XSwgWzUxLjIzNzkzNzkwMTE2OTA2LCAzLjM5NDA5MzM1NTE2MzkzMl0sIFs1MS4yMzc4ODk1Mzc1ODc3NiwgMy4zOTQxODY3Mjk1NjUyNjNdLCBbNTEuMjM3ODU0NTAxMjMyNTA1LCAzLjM5NDI5MTY3MDk5Mjk3MDVdLCBbNTEuMjM3ODU5NTMwMzc0NDEsIDMuMzk0NDE2NzI4OTg4MjldLCBbNTEuMjM3ODY2MjM1ODk2OTQ1LCAzLjM5NDUzNTA4MTQ2MTA3Ml0sIFs1MS4yMzc4OTEyMTM5NjgzOTYsIDMuMzk0NjgwMDA0NTY2OTA4XSwgWzUxLjIzNzg5NDU2NjcyOTY2NSwgMy4zOTQ3OTAwNTg5NTU1NV0sIFs1MS4yMzc4Njk1MDQ4MzkxOCwgMy4zOTQ4ODE2NzMxNTcyMTVdXSwKICAgICAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJvcmFuZ2UiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IGZhbHNlLAogICJmaWxsQ29sb3IiOiAib3JhbmdlIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJub0NsaXAiOiBmYWxzZSwKICAib3BhY2l0eSI6IDEuMCwKICAic21vb3RoRmFjdG9yIjogMS4wLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAgICAgLmFkZFRvKG1hcF84MjQxMmFmNTRiNTc0ZTEzYmJlNDMyMDFkZjFmMTczNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgICAgICB2YXIgcG9seV9saW5lXzdiMjlkYTFhYjRlNTQyZTNhZmJjNjRlMzJjOGE5ODRhID0gTC5wb2x5bGluZSgKICAgICAgICAgICAgICAgICAgICBbWzUxLjIzNzg0NDUyNjc2NzczLCAzLjM5NDg3ODQwNDIxNDk3OF0sIFs1MS4yMzc4MTI4NDMxNzM3NCwgMy4zOTQ5NjE3MjAzMzI1MDMzXSwgWzUxLjIzNzc3NDUzNzg3NjI1LCAzLjM5NTAyNjY4MDA4MjA4MjddLCBbNTEuMjM3Njg3ODY4OTk3NDU1LCAzLjM5NTAxNTAyOTIzNjY3NDNdLCBbNTEuMjM3NjExMTc0NTgzNDM1LCAzLjM5NTAzMzM4NTYwNDYyXSwgWzUxLjIzNzUyOTUzNDg0NjU0NCwgMy4zOTUwNDE2ODM2ODg3Nl0sIFs1MS4yMzc0NDI4NjU5Njc3NSwgMy4zOTUwNTE3NDE5NzI1NjU3XSwgWzUxLjIzNzM1Mjg0NDMyNzY5LCAzLjM5NTA2Njc0NTU3OTI0MjddLCBbNTEuMjM3MjYyOTA2NTA2NjYsIDMuMzk1MDc2NzIwMDQ0MDE3XSwgWzUxLjIzNzE3NzgzMDE4OTQ3LCAzLjM5NTA4NTAxODEyODE1NjddLCBbNTEuMjM3MDkyODM3NjkxMzEsIDMuMzk1MDg4MzcwODg5NDI1M10sIFs1MS4yMzcwMDk1MjE1NzM3OCwgMy4zOTUwOTE3MjM2NTA2OTRdLCBbNTEuMjM2OTI5NTU4MjE3NTI1LCAzLjM5NTA4ODM3MDg4OTQyNTNdLCBbNTEuMjM2ODU2MjE2NTY0Nzc1LCAzLjM5NTA4MDA3MjgwNTI4NTVdLCBbNTEuMjM2NzkxMTcyOTk2MTYsIDMuMzk1MDgzMzQxNzQ3NTIyNF0sIFs1MS4yMzY3MzExNTg1Njk0NTUsIDMuMzk1MDk4MzQ1MzU0MTk5NF0sIFs1MS4yMzY3MzQ1MTEzMzA3MjQsIDMuMzk1MTUwMDYxNjk2NzY4XSwgWzUxLjIzNjczMTE1ODU2OTQ1NSwgMy4zOTUxOTMzOTYxMzYxNjQ3XSwgWzUxLjIzNjczNjE4NzcxMTM2LCAzLjM5NTIzNTA1NDE5NDkyN10sIFs1MS4yMzY3ODQ1NTEyOTI2NiwgMy4zOTUyNDUwMjg2NTk3MDEzXSwgWzUxLjIzNjgzMjgzMTA1NDkyNiwgMy4zOTUyNDUwMjg2NTk3MDEzXSwgWzUxLjIzNjg4MTE5NDYzNjIyNiwgMy4zOTUyMzg0MDY5NTYxOTZdLCBbNTEuMjM2OTQyODg1NDQzNTcsIDMuMzk1MjI4MzQ4NjcyMzldLCBbNTEuMjM3MDExMTk3OTU0NDE2LCAzLjM5NTIxODM3NDIwNzYxNl0sIFs1MS4yMzcwODc4OTIzNjg0MzYsIDMuMzk1MjAxNjk0MjIwMzA0NV0sIFs1MS4yMzcxNjEyMzQwMjExOSwgMy4zOTUxNzgzOTI1Mjk0ODc2XSwgWzUxLjIzNzIzOTUyMDk5NjgxLCAzLjM5NTE2MTcxMjU0MjE3NjJdLCBbNTEuMjM3MzE2MjE1NDEwODMsIDMuMzk1MTQ1MDMyNTU0ODY1XSwgWzUxLjIzNzM5MjgyNjAwNTgyLCAzLjM5NTEyNjY3NjE4NjkxOV0sIFs1MS4yMzc0NjYxNjc2NTg1NywgMy4zOTUxMTAwODAwMTg2Mzk2XSwgWzUxLjIzNzUzNzgzMjkzMDY4NCwgMy4zOTUwOTE3MjM2NTA2OTRdLCBbNTEuMjM3NjExMTc0NTgzNDM1LCAzLjM5NTA3MzM2NzI4Mjc0ODJdLCBbNTEuMjM3Njg2MTkyNjE2ODIsIDMuMzk1MDU2Njg3Mjk1NDM3XSwgWzUxLjIzNzc2Mjg4NzAzMDg0LCAzLjM5NTA0MDAwNzMwODEyNTVdLCBbNTEuMjM3ODM3OTA1MDY0MjI1LCAzLjM5NTAyMDA1ODM3ODU3NzJdLCBbNTEuMjM3OTA3ODkzOTU1NzEsIDMuMzk0OTkxNzI3NTQ1ODU3NF0sIFs1MS4yMzc5NzI4NTM3MDUyOSwgMy4zOTQ5NTUwMTQ4MDk5NjZdLCBbNTEuMjM4MDQxMTY2MjE2MTM1LCAzLjM5NDkyMzQxNTAzNTAwOTRdLCBbNTEuMjM4MTA3ODg2MTY1MzgsIDMuMzk0ODk1MDAwMzgzMjU4XSwgWzUxLjIzODE3MTE2OTUzNDMyNiwgMy4zOTQ4NzAwMjIzMTE4MDY3XSwgWzUxLjIzODIyNzgzMTE5OTc2NSwgMy4zOTQ4MzUwNjk3NzU1ODE0XSwgWzUxLjIzODI2Mjg2NzU1NTAyLCAzLjM5NDc3NjczMTcyOTUwNzRdLCBbNTEuMjM4Mjc5NTQ3NTQyMzM0LCAzLjM5NDY5ODM2MDkzNDg1MzZdLCBbNTEuMjM4MjY5NTczMDc3NTYsIDMuMzk0NjA1MDcwMzUyNTU0M10sIFs1MS4yMzgyNjI4Njc1NTUwMiwgMy4zOTQ1MTY3MjUwOTMxMjYzXSwgWzUxLjIzODI1Mjg5MzA5MDI1LCAzLjM5NDQyMDA4MTc0OTU1ODRdLCBbNTEuMjM4MjU3ODM4NDEzMTIsIDMuMzk0MzMwMDYwMTA5NDk2XSwgWzUxLjIzODI2OTU3MzA3NzU2LCAzLjM5NDI1MTY4OTMxNDg0MjJdLCBbNTEuMjM4MzAyODQ5MjMzMTUsIDMuMzk0MjAzNDA5NTUyNTc0XSwgWzUxLjIzODM3MjgzODEyNDYzLCAzLjM5NDIxMzM4NDAxNzM0ODNdLCBbNTEuMjM4NDIyODc4MDg2NTcsIDMuMzk0MjQwMDM4NDY5NDM0XSwgWzUxLjIzODQ1NjIzODA2MTE5LCAzLjM5NDMxNjczMjg4MzQ1MzRdLCBbNTEuMjM4NDc3ODYzMzcxMzcsIDMuMzk0NDA1MDc4MTQyODgxNF0sIFs1MS4yMzg0ODI4OTI1MTMyNzUsIDMuMzk0NTMzNDA1MDgwNDM3N10sIFs1MS4yMzg0ODYxNjE0NTU1MSwgMy4zOTQ2NTMzNTAxMTQ4MjI0XSwgWzUxLjIzODQ5MTE5MDU5NzQxNSwgMy4zOTQ3NjUwODA4ODQwOTldLCBbNTEuMjM4NDkxMTkwNTk3NDE1LCAzLjM5NDg2MzQwMDYwODMwMV0sIFs1MS4yMzg0OTk1NzI1MDA1OSwgMy4zOTQ5NDMzNjM5NjQ1NTc2XSwgWzUxLjIzODUwNzg3MDU4NDcyNiwgMy4zOTQ5NzAwMTg0MTY2NDNdLCBbNTEuMjM4NTIxMTk3ODEwNzcsIDMuMzk0OTU1MDE0ODA5OTY2XSwgWzUxLjIzODUyOTQ5NTg5NDkxLCAzLjM5NDkyMTczODY1NDM3NV0sIFs1MS4yMzg1Mjc5MDMzMzMzMDYsIDMuMzk0ODczMzc1MDczMDc1M10sIFs1MS4yMzg1MDc4NzA1ODQ3MjYsIDMuMzk0Nzg4MzgyNTc0OTE2XSwgWzUxLjIzODQ5MTE5MDU5NzQxNSwgMy4zOTQ3MDUwNjY0NTczOTFdLCBbNTEuMjM4NDc0NTEwNjEwMTA0LCAzLjM5NDYxMzM2ODQzNjY5NF0sIFs1MS4yMzg0NjExODMzODQwNiwgMy4zOTQ1MTAwMTk1NzA1ODldLCBbNTEuMjM4NDQ3ODU2MTU4MDIsIDMuMzk0NDEwMDIzNDY1NzUyNl0sIFs1MS4yMzg0MzI4NTI1NTEzNCwgMy4zOTQzMTAwMjczNjA5MTZdLCBbNTEuMjM4NDI2MjMwODQ3ODM2LCAzLjM5NDIxODQxMzE1OTI1MV0sIFs1MS4yMzg0Mjk0OTk3OTAwNywgMy4zOTQxMzUwMTMyMjI2OTQ0XSwgWzUxLjIzODQ1NDU2MTY4MDU1NSwgMy4zOTQxMDE3MzcwNjcxMDM0XSwgWzUxLjIzODQ3MjgzNDIyOTQ3LCAzLjM5NDEwMTczNzA2NzEwMzRdLCBbNTEuMjM4NDgyODkyNTEzMjc1LCAzLjM5NDEzNjY4OTYwMzMyODddLCBbNTEuMjM4NDg5NTE0MjE2NzgsIDMuMzk0MTkxNjc0ODg4MTM0XSwgWzUxLjIzODQ5NjIxOTczOTMyLCAzLjM5NDI2NjY5MjkyMTUxOTNdLCBbNTEuMjM4NDk5NTcyNTAwNTksIDMuMzk0MzUxNjg1NDE5Njc4N10sIFs1MS4yMzg1MDQ1MTc4MjM0NiwgMy4zOTQ0NDg0MTI1ODIyNzgzXSwgWzUxLjIzODUxMTIyMzM0NTk5NSwgMy4zOTQ1NTUwMzAzOTA2MjAyXSwgWzUxLjIzODUyMTE5NzgxMDc3LCAzLjM5NDY3MTcwNjQ4Mjc2OF0sIFs1MS4yMzg1MjYyMjY5NTI2NywgMy4zOTQ3NjUwODA4ODQwOTldLCBbNTEuMjM4NTMxMTcyMjc1NTQsIDMuMzk0ODM1MDY5Nzc1NTgxNF0sIFs1MS4yMzg1MzQ1MjUwMzY4MSwgMy4zOTQ5MDE3MDU5MDU3OTVdLCBbNTEuMjM4NTQ3ODUyMjYyODU1LCAzLjM5NDkzMzM4OTQ5OTc4MzVdLCBbNTEuMjM4NTYyODU1ODY5NTMsIDMuMzk0OTE1MDMzMTMxODM4XSwgWzUxLjIzODU3MjgzMDMzNDMwNiwgMy4zOTQ4ODMzNDk1Mzc4NDk0XSwgWzUxLjIzODU2NDUzMjI1MDE2NiwgMy4zOTQ4MTY3MTM0MDc2MzU3XSwgWzUxLjIzODU1Mjg4MTQwNDc2LCAzLjM5NDc0ODQwMDg5Njc4NzZdLCBbNTEuMjM4NTQ3ODUyMjYyODU1LCAzLjM5NDY2MTczMjAxNzk5NF0sIFs1MS4yMzg1NjI4NTU4Njk1MywgMy4zOTQ1NjM0MTIyOTM3OTE4XSwgWzUxLjIzODU5Mjg2MzA4Mjg4NiwgMy4zOTQ0NzY3NDM0MTQ5OThdLCBbNTEuMjM4NjM0NTIxMTQxNjUsIDMuMzk0NDAxNzI1MzgxNjEyOF0sIFs1MS4yMzg2NzEyMzM4Nzc1NCwgMy4zOTQzMjE2NzgyMDYzMjQ2XSwgWzUxLjIzODcwNjE4NjQxMzc2NSwgMy4zOTQyMzUwMDkzMjc1MzFdLCBbNTEuMjM4NzM5NTQ2Mzg4MzksIDMuMzk0MTQxNzE4NzQ1MjMxNl0sIFs1MS4yMzg3NzEyMjk5ODIzNzYsIDMuMzk0MDQ2NjY3OTYzMjY2NF0sIFs1MS4yMzg4MDEyMzcxOTU3MywgMy4zOTM5NTg0MDY1MjI4N10sIFs1MS4yMzg4MzQ1MTMzNTEzMiwgMy4zOTM4ODAwMzU3MjgyMTZdLCBbNTEuMjM4ODc0NDk1MDI5NDUsIDMuMzkzODM1MDI0OTA4MTg1XSwgWzUxLjIzODkwNzg1NTAwNDA3LCAzLjM5MzgwMTc0ODc1MjU5NF0sIFs1MS4yMzg5NDI4OTEzNTkzMywgMy4zOTM3OTE2OTA0Njg3ODhdLCBbNTEuMjM4OTcyODk4NTcyNjgsIDMuMzkzNzg1MDY4NzY1MjgyNl0sIFs1MS4yMzg5OTExNzExMjE2LCAzLjM5Mzc5MDAxNDA4ODE1NF0sIFs1MS4yMzg5NDk1MTMwNjI4MzUsIDMuMzkzODYzMzU1NzQwOTA1XSwgWzUxLjIzODkwOTUzMTM4NDcxLCAzLjM5Mzk0NjY3MTg1ODQzXSwgWzUxLjIzODg1NDU0NjA5OTksIDMuMzk0MDUwMDIwNzI0NTM1XSwgWzUxLjIzODgxNjE1Njk4MzM3NiwgMy4zOTQxNTE2OTMyMTAwMDU4XSwgWzUxLjIzODc3NjE3NTMwNTI1LCAzLjM5NDI1NTA0MjA3NjExMV0sIFs1MS4yMzg3MzQ1MTcyNDY0ODUsIDMuMzk0MzQ4MzMyNjU4NDFdLCBbNTEuMjM4NjkyODU5MTg3NzIsIDMuMzk0NDM4MzU0Mjk4NDcyNF0sIFs1MS4yMzg2NTc5MDY2NTE1LCAzLjM5NDUzODM1MDQwMzMwOV0sIFs1MS4yMzg2NDI5MDMwNDQ4MiwgMy4zOTQ2NDMzNzU2NTAwNDgzXSwgWzUxLjIzODY0NjE3MTk4NzA2LCAzLjM5NDc1NTAyMjYwMDI5M10sIFs1MS4yMzg2NjQ1MjgzNTUsIDMuMzk0ODc2NzI3ODM0MzQ0XSwgWzUxLjIzODY5OTU2NDcxMDI2LCAzLjM5NDk1MDA2OTQ4NzA5NV0sIFs1MS4yMzg3NjI4NDgwNzkyMDUsIDMuMzk1MDA4NDA3NTMzMTY5XSwgWzUxLjIzODgyMTE4NjEyNTI4LCAzLjM5NTAyMzQxMTEzOTg0Nl0sIFs1MS4yMzg4Nzk1MjQxNzEzNSwgMy4zOTUwMzMzODU2MDQ2Ml0sIFs1MS4yMzg5MjQ1MzQ5OTEzOCwgMy4zOTUwMTgzODE5OTc5NDNdLCBbNTEuMjM4OTYyODQwMjg4ODgsIDMuMzk0OTkwMDUxMTY1MjIzXSwgWzUxLjIzODk1Mjg2NTgyNDEsIDMuMzk0OTM4MzM0ODIyNjU0N10sIFs1MS4yMzg5MTk1MDU4NDk0OCwgMy4zOTQ4OTM0MDc4MjE2NTUzXSwgWzUxLjIzODg4NDU1MzMxMzI1NSwgMy4zOTQ4NDUwNDQyNDAzNTU1XSwgWzUxLjIzODg1Mjg2OTcxOTI3LCAzLjM5NDc4MzM1MzQzMzAxM10sIFs1MS4yMzg4Mzc4NjYxMTI1OSwgMy4zOTQ3MDgzMzUzOTk2Mjc3XSwgWzUxLjIzODgzNjE4OTczMTk1NiwgMy4zOTQ2MjgzNzIwNDMzNzFdLCBbNTEuMjM4ODYxMTY3ODAzNDEsIDMuMzk0NjEwMDE1Njc1NDI1NV0sIFs1MS4yMzg4ODYyMjk2OTM4OSwgMy4zOTQ1OTUwMTIwNjg3NDg1XSwgWzUxLjIzODkxNjIzNjkwNzI0NCwgMy4zOTQ1NzAwMzM5OTcyOTczXSwgWzUxLjIzODk0NDU2NzczOTk2LCAzLjM5NDUzMTcyODY5OTgwMzRdLCBbNTEuMjM4OTQ0NTY3NzM5OTYsIDMuMzk0NDcxNzE0MjczMDk1XSwgWzUxLjIzODk0MTIxNDk3ODY5NSwgMy4zOTQ0MTE2OTk4NDYzODddLCBbNTEuMjM4OTA0NTAyMjQyODA0LCAzLjM5NDM1MDAwOTAzOTA0NDRdLCBbNTEuMjM4ODgyODc2OTMyNjIsIDMuMzk0Mjc4MzQzNzY2OTI3N10sIFs1MS4yMzg4NzQ0OTUwMjk0NSwgMy4zOTQyMDAwNTY3OTEzMDU1XSwgWzUxLjIzODg3Nzg0Nzc5MDcyLCAzLjM5NDEyNTAzODc1NzkyMDNdLCBbNTEuMjM4ODk2MjA0MTU4NjY0LCAzLjM5NDA2MDA3OTAwODM0MV0sIFs1MS4yMzg5NDk1MTMwNjI4MzUsIDMuMzk0MDc4MzUxNTU3MjU1XSwgWzUxLjIzODk4Mjg3MzAzNzQ2LCAzLjM5NDEyNTAzODc1NzkyMDNdLCBbNTEuMjM4OTk2MjAwMjYzNSwgMy4zOTQxOTgzODA0MTA2NzEyXSwgWzUxLjIzOTAwNDQ5ODM0NzY0LCAzLjM5NDI5ODM3NjUxNTUwNzddLCBbNTEuMjM5MDA3ODUxMTA4OTEsIDMuMzk0NDEzMzc2MjI3MDIxXSwgWzUxLjIzOTAxNDU1NjYzMTQ0NiwgMy4zOTQ1MzAwNTIzMTkxNjldLCBbNTEuMjM5MDE3ODI1NTczNjgsIDMuMzk0NjQ2NzI4NDExMzE3XSwgWzUxLjIzOTAwOTUyNzQ4OTU0LCAzLjM5NDc2NjY3MzQ0NTcwMTZdLCBbNTEuMjM5MDExMjAzODcwMTgsIDMuMzk0ODczMzc1MDczMDc1M10sIFs1MS4yMzkwMjExNzgzMzQ5NSwgMy4zOTQ5NjE3MjAzMzI1MDMzXSwgWzUxLjIzOTA0NjI0MDIyNTQzNCwgMy4zOTUwMDY3MzExNTI1MzQ1XSwgWzUxLjIzOTA3NjE2MzYxOTc2LCAzLjM5NTAwNTA1NDc3MTldLCBbNTEuMjM5MTAyOTAxODkwODc0LCAzLjM5NDk4MTY2OTI2MjA1MTZdLCBbNTEuMjM5MTE3OTA1NDk3NTUsIDMuMzk0OTQwMDExMjAzMjg5XSwgWzUxLjIzOTEyNDUyNzIwMTA2LCAzLjM5NDg3NjcyNzgzNDM0NF0sIFs1MS4yMzkxMTYyMjkxMTY5MiwgMy4zOTQ3OTUwMDQyNzg0MjE0XSwgWzUxLjIzOTA5NjE5NjM2ODM0LCAzLjM5NDY5MzMzMTc5Mjk1MDZdLCBbNTEuMjM5MDk0NTE5OTg3NywgMy4zOTQ1OTMzMzU2ODgxMTRdLCBbNTEuMjM5MTAxMjI1NTEwMjQsIDMuMzk0NDg2NzE3ODc5NzcyXSwgWzUxLjIzOTEwMjkwMTg5MDg3NCwgMy4zOTQzODE2OTI2MzMwMzNdLCBbNTEuMjM5MTExMTk5OTc1MDE0LCAzLjM5NDI4MTY5NjUyODE5NjNdLCBbNTEuMjM5MTI5NTU2MzQyOTYsIDMuMzk0MTkzMzUxMjY4NzY4M10sIFs1MS4yMzkxNTQ1MzQ0MTQ0MSwgMy4zOTQxMTY3NDA2NzM3ODA0XSwgWzUxLjIzOTIwMjg5Nzk5NTcxLCAzLjM5NDA5MDAwMjQwMjY2MzJdLCBbNTEuMjM5MjYyODI4NjAzMzksIDMuMzk0MTE1MDY0MjkzMTQ2XSwgWzUxLjIzOTMwNzgzOTQyMzQyLCAzLjM5NDE2NTAyMDQzNjA0ODVdLCBbNTEuMjM5MzM0NDkzODc1NSwgMy4zOTQyMzUwMDkzMjc1MzFdLCBbNTEuMjM5MzI3ODcyMTcyLCAzLjM5NDMxODQwOTI2NDA4NzddLCBbNTEuMjM5MzA2MTYzMDQyNzg0LCAzLjM5NDQwODM0NzA4NTExODNdLCBbNTEuMjM5Mjc5NTA4NTkwNywgMy4zOTQ0OTUwMTU5NjM5MTJdLCBbNTEuMjM5MjU2MjA2ODk5ODgsIDMuMzk0NTg1MDM3NjAzOTc0M10sIFs1MS4yMzkyNDc4MjQ5OTY3MSwgMy4zOTQ2ODMzNTczMjgxNzY1XSwgWzUxLjIzOTI1NDUzMDUxOTI1LCAzLjM5NDc3ODQwODExMDE0MThdLCBbNTEuMjM5MjU3ODgzMjgwNTE2LCAzLjM5NDg2MzQwMDYwODMwMV0sIFs1MS4yMzkyNjI4Mjg2MDMzOSwgMy4zOTQ5MzMzODk0OTk3ODM1XSwgWzUxLjIzOTI3OTUwODU5MDcsIDMuMzk0OTUxNzQ1ODY3NzI5XSwgWzUxLjIzOTI5Nzg2NDk1ODY0NCwgMy4zOTQ5MzY3NDIyNjEwNTJdLCBbNTEuMjM5Mjk2MTg4NTc4MDEsIDMuMzk0ODk2Njc2NzYzODkyXSwgWzUxLjIzOTI3NDU2MzI2NzgzLCAzLjM5NDgyNjY4Nzg3MjQxXSwgWzUxLjIzOTI1OTU1OTY2MTE1LCAzLjM5NDc1MTY2OTgzOTAyNDVdLCBbNTEuMjM5MjQ5NTAxMzc3MzQ0LCAzLjM5NDY2MzQwODM5ODYyODJdLCBbNTEuMjM5MjY2MTgxMzY0NjU1LCAzLjM5NDU1ODM4MzE1MTg4OV0sIFs1MS4yMzkyODI4NjEzNTE5NywgMy4zOTQ0NjAwNjM0Mjc2ODY3XSwgWzUxLjIzOTMxMjg2ODU2NTMyLCAzLjM5NDM3NTA3MDkyOTUyNzNdLCBbNTEuMjM5Mzg0NTMzODM3NDQsIDMuMzk0MzY1MDEyNjQ1NzIxNF0sIFs1MS4yMzk0MzYxNjYzNjA5NzQsIDMuMzk0MzgzMzY5MDEzNjY3XSwgWzUxLjIzOTQ3MTIwMjcxNjIzLCAzLjM5NDQyNTAyNzA3MjQyOTddLCBbNTEuMjM5NDcxMjAyNzE2MjMsIDMuMzk0NTAzMzk3ODY3MDgzNV0sIFs1MS4yMzk0NTk1NTE4NzA4MiwgMy4zOTQ1ODMzNjEyMjMzNF0sIFs1MS4yMzk0Mzk1MTkxMjIyNCwgMy4zOTQ2NzE3MDY0ODI3NjhdLCBbNTEuMjM5NDI3ODY4Mjc2ODM0LCAzLjM5NDc2NTA4MDg4NDA5OV0sIFs1MS4yMzk0MzEyMjEwMzgxLCAzLjM5NDg2MDA0Nzg0NzAzMjVdLCBbNTEuMjM5NDQ5NDkzNTg3MDIsIDMuMzk0OTM2NzQyMjYxMDUyXSwgWzUxLjIzOTQ4Nzg4MjcwMzU0LCAzLjM5NTAwMTcwMjAxMDYzMTZdLCBbNTEuMjM5NTM0NTY5OTA0MjEsIDMuMzk1MDIxNzM0NzU5MjExNV0sIFs1MS4yMzk1ODExNzMyODU4NCwgMy4zOTUwMDMzNzgzOTEyNjZdLCBbNTEuMjM5NjE2MjA5NjQxMSwgMy4zOTQ5Nzg0MDAzMTk4MTQ3XSwgWzUxLjIzOTY1MjgzODU1Nzk2LCAzLjM5NDkzODMzNDgyMjY1NDddLCBbNTEuMjM5NjM2MTU4NTcwNjUsIDMuMzk0ODI4MzY0MjUzMDQ0XSwgWzUxLjIzOTYzMTIxMzI0Nzc3NiwgMy4zOTQ3MzMzOTcyOTAxMTA2XSwgWzUxLjIzOTYzMjg4OTYyODQxLCAzLjM5NDY0MzM3NTY1MDA0ODNdLCBbNTEuMjM5NjQyODY0MDkzMTg0LCAzLjM5NDU2MzQxMjI5Mzc5MThdLCBbNTEuMjM5NjUxMTYyMTc3MzI0LCAzLjM5NDQ4NjcxNzg3OTc3Ml0sIFs1MS4yMzk2NTQ1MTQ5Mzg1OSwgMy4zOTQ0MTUwNTI2MDc2NTU1XSwgWzUxLjIzOTY1NDUxNDkzODU5LCAzLjM5NDM0MzM4NzMzNTUzOV0sIFs1MS4yMzk2NTExNjIxNzczMjQsIDMuMzk0Mjc2NjY3Mzg2MjkzNF0sIFs1MS4yMzk2NDk1Njk2MTU3MiwgMy4zOTQyMTUwNjAzOTc5ODI2XSwgWzUxLjIzOTY0Nzg5MzIzNTA5LCAzLjM5NDE1MDAxNjgyOTM3MTVdLCBbNTEuMjM5NjQ2MjE2ODU0NDUsIDMuMzk0MDc2Njc1MTc2NjIwNV0sIFs1MS4yMzk2NDExODc3MTI1NSwgMy4zOTM5ODUwNjA5NzQ5NTU2XSwgWzUxLjIzOTYzNjE1ODU3MDY1LCAzLjM5Mzg5NjcxNTcxNTUyNzVdLCBbNTEuMjM5NjIyODMxMzQ0NjA0LCAzLjM5Mzc5NjcxOTYxMDY5MV0sIFs1MS4yMzk2MTQ1MzMyNjA0NjUsIDMuMzkzNzAxNjY4ODI4NzI2XSwgWzUxLjIzOTYxMTE4MDQ5OTE5NiwgMy4zOTM2MjE3MDU0NzI0NjkzXSwgWzUxLjIzOTYyMjgzMTM0NDYwNCwgMy4zOTM1MzgzODkzNTQ5NDQyXSwgWzUxLjIzOTY0NDU0MDQ3MzgyLCAzLjM5MzQ3NjY5ODU0NzYwMTddLCBbNTEuMjM5NzA5NTAwMjIzNCwgMy4zOTM0NzUwMjIxNjY5Njc0XSwgWzUxLjIzOTc2Mjg5Mjk0NjYsIDMuMzkzNDU1MDczMjM3NDE5XSwgWzUxLjIzOTgyNDQ5OTkzNDkxLCAzLjM5MzQzMzM2NDEwODIwNV0sIFs1MS4yMzk4OTEyMTk4ODQxNiwgMy4zOTM0MTE3Mzg3OTgwMjIzXSwgWzUxLjIzOTk2Mjg4NTE1NjI3NCwgMy4zOTMzOTUwNTg4MTA3MTFdLCBbNTEuMjQwMDM0NTUwNDI4MzksIDMuMzkzMzg1MDAwNTI2OTA1XSwgWzUxLjI0MDEwNDUzOTMxOTg3LCAzLjM5MzM3NTAyNjA2MjEzMV0sIFs1MS4yNDAxNzI4NTE4MzA3MiwgMy4zOTMzNzAwODA3MzkyNTk3XSwgWzUxLjI0MDI0NDUxNzEwMjg0LCAzLjM5MzM3MDA4MDczOTI1OTddLCBbNTEuMjQwMzEyODI5NjEzNjg2LCAzLjM5MzM3MTY3MzMwMDg2MjNdLCBbNTEuMjQwMzgxMjI1OTQzNTY1LCAzLjM5MzM3MTY3MzMwMDg2MjNdLCBbNTEuMjQwNDQ3ODYyMDczNzgsIDMuMzkzMzY2NzI3OTc3OTkxXSwgWzUxLjI0MDUxNDQ5ODIwMzk5LCAzLjM5MzM1ODM0NjA3NDgxOTZdLCBbNTEuMjQwNTc0NTEyNjMwNywgMy4zOTMzNDUwMTg4NDg3NzddLCBbNTEuMjQwNjMyODUwNjc2Nzc1LCAzLjM5MzMyMzM5MzUzODU5NDJdLCBbNTEuMjQwNjgyODkwNjM4NzEsIDMuMzkzMjk1MDYyNzA1ODc0NF0sIFs1MS4yNDA3MjYyMjUwNzgxMDYsIDMuMzkzMjczMzUzNTc2NjZdLCBbNTEuMjQwNzQxMjI4Njg0NzgsIDMuMzkzMjMzMzcxODk4NTMyXSwgWzUxLjI0MDY2Mjg1Nzg5MDEzLCAzLjM5MzE1MTczMjE2MTY0MV0sIFs1MS4yNDA2MTI5MDE3NDcyMywgMy4zOTMxMzg0MDQ5MzU1OTg0XSwgWzUxLjI0MDU1NDU2MzcwMTE1LCAzLjM5MzEyMDA0ODU2NzY1MjddLCBbNTEuMjQwNDk0NTQ5Mjc0NDQ1LCAzLjM5MzA4NjY4ODU5MzAzXSwgWzUxLjI0MDQ0NjE4NTY5MzE0NSwgMy4zOTMwNDAwMDEzOTIzNjQ1XSwgWzUxLjI0MDQ1MTIxNDgzNTA1LCAzLjM5Mjk0NTAzNDQyOTQzMV0sIFs1MS4yNDA0NDYxODU2OTMxNDUsIDMuMzkyODY2NzQ3NDUzODA5XSwgWzUxLjI0MDQ0OTUzODQ1NDQxLCAzLjM5MjgxODM4Mzg3MjUwOV0sIFs1MS4yNDA0Nzk1NDU2Njc3NywgMy4zOTI4MzAwMzQ3MTc5MTc0XSwgWzUxLjI0MDUwMjg0NzM1ODU4NCwgMy4zOTI4NTE3NDM4NDcxMzE3XSwgWzUxLjI0MDUzOTU2MDA5NDQ3NiwgMy4zOTI4ODUwMjAwMDI3MjI3XSwgWzUxLjI0MDU2MTE4NTQwNDY2LCAzLjM5Mjg4NjY5NjM4MzM1N10sIFs1MS4yNDA1Nzc4NjUzOTE5NywgMy4zOTI4NTY2ODkxNzAwMDNdLCBbNTEuMjQwNTcyODM2MjUwMDcsIDMuMzkyNzk2Njc0NzQzMjk0N10sIFs1MS4yNDA1NjYyMTQ1NDY1NiwgMy4zOTI3MzMzOTEzNzQzNDk2XSwgWzUxLjI0MDU1NDU2MzcwMTE1LCAzLjM5MjY3MzM3Njk0NzY0MTRdLCBbNTEuMjQwNTIyODgwMTA3MTY0LCAzLjM5MjY2MTcyNjEwMjIzM10sIFs1MS4yNDA0ODEyMjIwNDg0LCAzLjM5MjYzODM0MDU5MjM4NDNdLCBbNTEuMjQwNDIxMjA3NjIxNjk0LCAzLjM5MjY4MzM1MTQxMjQxNTVdLCBbNTEuMjQwMzY2MjIyMzM2ODksIDMuMzkyNzAzMzg0MTYwOTk1NV0sIFs1MS4yNDAzMDc4ODQyOTA4MTQsIDMuMzkyNzE1MDM1MDA2NDA0XSwgWzUxLjI0MDI1MTIyMjYyNTM3NSwgMy4zOTI3MjY2ODU4NTE4MTI0XSwgWzUxLjI0MDE4MTIzMzczMzg5LCAzLjM5Mjc0ODM5NDk4MTAyNjZdLCBbNTEuMjQwMTEyODM3NDA0MDEsIDMuMzkyNzY1MDc0OTY4MzM4XSwgWzUxLjI0MDA1MTIzMDQxNTcsIDMuMzkyNzg1MDIzODk3ODg2M10sIFs1MS4yMzk5Nzk1NjUxNDM1ODUsIDMuMzkyODAwMDI3NTA0NTYzM10sIFs1MS4yMzk5MTc4NzQzMzYyNCwgMy4zOTI4MDAwMjc1MDQ1NjMzXSwgWzUxLjIzOTg2NDU2NTQzMjA3LCAzLjM5MjgwNTA1NjY0NjQ2NjNdLCBbNTEuMjM5ODIyOTA3MzczMzEsIDMuMzkyODMxNzExMDk4NTUxOF0sIFs1MS4yMzk4MTc4NzgyMzE0MDYsIDMuMzkyODUwMDY3NDY2NDk3NF0sIFs1MS4yMzk4MTk1NTQ2MTIwNCwgMy4zOTI4NzUwNDU1Mzc5NDg2XSwgWzUxLjIzOTg0OTU2MTgyNTM5NSwgMy4zOTI4ODgzNzI3NjM5OTE0XSwgWzUxLjIzOTg5Nzg0MTU4NzY2LCAzLjM5Mjg5MDA0OTE0NDYyNTddLCBbNTEuMjM5OTQ0NTI4Nzg4MzMsIDMuMzkyODc4Mzk4Mjk5MjE3Ml0sIFs1MS4yNDAwMDExOTA0NTM3NywgMy4zOTI4NjMzOTQ2OTI1NF0sIFs1MS4yNDAwNTEyMzA0MTU3LCAzLjM5Mjg0NjcxNDcwNTIyOV0sIFs1MS4yNDAxMDYyMTU3MDA1MSwgMy4zOTI4MzAwMzQ3MTc5MTc0XSwgWzUxLjI0MDE2OTQ5OTA2OTQ1LCAzLjM5MjgxNTAzMTExMTI0MDRdLCBbNTEuMjQwMjI2MTYwNzM0ODksIDMuMzkyODAwMDI3NTA0NTYzM10sIFs1MS4yNDAzMDQ1MzE1Mjk1NDYsIDMuMzkyNzcxNjk2NjcxODQzNV0sIFs1MS4yNDAzNjc4OTg3MTc1MiwgMy4zOTI3NDMzNjU4MzkxMjM3XSwgWzUxLjI0MDQ0MTI0MDM3MDI3NCwgMy4zOTI3MjY2ODU4NTE4MTI0XSwgWzUxLjI0MDUwNzg3NjUwMDQ5LCAzLjM5MjcyNTAwOTQ3MTE3OF0sIFs1MS4yNDA1NjI4NjE3ODUyOSwgMy4zOTI3NDAwMTMwNzc4NTVdLCBbNTEuMjQwNTg5NTE2MjM3MzgsIDMuMzkyODEzMzU0NzMwNjA2XSwgWzUxLjI0MDU4OTUxNjIzNzM4LCAzLjM5Mjg2NTA3MTA3MzE3NDVdLCBbNTEuMjQwNTQ0NTA1NDE3MzUsIDMuMzkyODUzMzM2NDA4NzM0M10sIFs1MS4yNDA1MjQ1NTY0ODc4LCAzLjM5Mjg1MzMzNjQwODczNDNdLCBbNTEuMjQwNTA0NTIzNzM5MjIsIDMuMzkyODE2NzA3NDkxODc0N10sIFs1MS4yNDA0NzYxOTI5MDY1LCAzLjM5Mjc4NTAyMzg5Nzg4NjNdLCBbNTEuMjQwNDQxMjQwMzcwMjc0LCAzLjM5Mjc3MDAyMDI5MTIwOV0sIFs1MS4yNDAzODI5MDIzMjQyLCAzLjM5Mjc4MzM0NzUxNzI1Ml0sIFs1MS4yNDAzMjQ1NjQyNzgxMjYsIDMuMzkyODA2NzMzMDI3MTAwNl0sIFs1MS4yNDAyNjI4NzM0NzA3OCwgMy4zOTI4MjM0MTMwMTQ0MTJdLCBbNTEuMjQwMTk5NTA2MjgyODA2LCAzLjM5Mjg0NjcxNDcwNTIyOV0sIFs1MS4yNDAxMzYyMjI5MTM4NiwgMy4zOTI4NjY3NDc0NTM4MDldLCBbNTEuMjQwMDcxMTc5MzQ1MjUsIDMuMzkyODg1MDIwMDAyNzIyN10sIFs1MS4yNDAwMDYyMTk1OTU2NywgMy4zOTI4OTgzNDcyMjg3NjU1XSwgWzUxLjIzOTk0NDUyODc4ODMzLCAzLjM5MjkxMDA4MTg5MzIwNTZdLCBbNTEuMjM5ODg3ODY3MTIyODksIDMuMzkyOTIwMDU2MzU3OTc5OF0sIFs1MS4yMzk4MzYyMzQ1OTkzNSwgMy4zOTI5MzAwMzA4MjI3NTRdLCBbNTEuMjM5Nzk0NDkyNzIxNTYsIDMuMzkyOTQ1MDM0NDI5NDMxXSwgWzUxLjIzOTc4OTU0NzM5ODY4NiwgMy4zOTI5ODE3NDcxNjUzMjIzXSwgWzUxLjIzOTgxNDUyNTQ3MDE0LCAzLjM5Mjk4MzMzOTcyNjkyNV0sIFs1MS4yMzk4NDc4ODU0NDQ3NiwgMy4zOTI5ODE3NDcxNjUzMjIzXSwgWzUxLjIzOTg5Mjg5NjI2NDc5LCAzLjM5Mjk2Njc0MzU1ODY0NTJdLCBbNTEuMjM5OTQyODUyNDA3Njk0LCAzLjM5Mjk0ODM4NzE5MDY5OTZdLCBbNTEuMjQwMDA2MjE5NTk1NjcsIDMuMzkyOTMxNzA3MjAzMzg4XSwgWzUxLjI0MDA4MTIzNzYyOTA1NiwgMy4zOTI5MjE3MzI3Mzg2MTRdLCBbNTEuMjQwMTU2MTcxODQzNDEsIDMuMzkyOTAzMzc2MzcwNjY4NF0sIFs1MS4yNDAyMjEyMTU0MTIwMiwgMy4zOTI4ODgzNzI3NjM5OTE0XSwgWzUxLjI0MDI3Nzg3NzA3NzQ2LCAzLjM5Mjg4NjY5NjM4MzM1N10sIFs1MS4yNDAzMDExNzg3NjgyOCwgMy4zOTI5MTE2NzQ0NTQ4MDgyXSwgWzUxLjI0MDMwNDUzMTUyOTU0NiwgMy4zOTI5NTUwMDg4OTQyMDVdLCBbNTEuMjQwMjk0NTU3MDY0NzcsIDMuMzkyOTkwMDQ1MjQ5NDYyXSwgWzUxLjI0MDI3NDUyNDMxNjE5LCAzLjM5MzAyMDA1MjQ2MjgxNjJdLCBbNTEuMjQwMjA2MjExODA1MzQ0LCAzLjM5MzAyMDA1MjQ2MjgxNjJdLCBbNTEuMjQwMTQxMTY4MjM2NzMsIDMuMzkzMDIwMDUyNDYyODE2Ml0sIFs1MS4yNDAwNjk1MDI5NjQ2MTYsIDMuMzkzMDI4MzUwNTQ2OTU2XSwgWzUxLjI0MDAwNjIxOTU5NTY3LCAzLjM5MzA0MTY3Nzc3Mjk5OV0sIFs1MS4yMzk5NTI4MjY4NzI0NywgMy4zOTMwNjMzODY5MDIyMTNdLCBbNTEuMjM5OTA5NDkyNDMzMDcsIDMuMzkzMDgwMDY2ODg5NTI0NV0sIFs1MS4yMzk5MjI5MDM0NzgxNDYsIDMuMzkzMTEwMDc0MTAyODc4Nl0sIFs1MS4yMzk5NDk1NTc5MzAyMywgMy4zOTMxNDUwMjY2MzkxMDRdLCBbNTEuMjM5OTcyODU5NjIxMDUsIDMuMzkzMTYwMDMwMjQ1NzgxXSwgWzUxLjI0MDAyNjE2ODUyNTIyLCAzLjM5MzEzMTY5OTQxMzA2MV0sIFs1MS4yNDAwODEyMzc2MjkwNTYsIDMuMzkzMTAzMzY4NTgwMzQxM10sIFs1MS4yNDAxNDc4NzM3NTkyNywgMy4zOTMwNzY3MTQxMjgyNTZdLCBbNTEuMjQwMjA5NTY0NTY2NjEsIDMuMzkzMDg1MDEyMjEyMzk1N10sIFs1MS4yNDAyNzQ1MjQzMTYxOSwgMy4zOTMxMDgzOTc3MjIyNDQzXSwgWzUxLjI0MDMzNjIxNTEyMzUzNCwgMy4zOTMxNDMzNTAyNTg0Njk2XSwgWzUxLjI0MDM5NjIyOTU1MDI0LCAzLjM5MzE3MDAwNDcxMDU1NV0sIFs1MS4yNDA0NTI4OTEyMTU2OCwgMy4zOTMxOTUwNjY2MDEwMzhdLCBbNTEuMjQwNDcxMTYzNzY0NTk2LCAzLjM5MzI0NjY5OTEyNDU3NDddLCBbNTEuMjQwNDUxMjE0ODM1MDUsIDMuMzkzMjYxNzAyNzMxMjUxN10sIFs1MS4yNDA0MzQ1MzQ4NDc3MzYsIDMuMzkzMjc2NzA2MzM3OTI4OF0sIFs1MS4yNDAzODYxNzEyNjY0NCwgMy4zOTMyNDE2Njk5ODI2NzE3XSwgWzUxLjI0MDMzOTU2Nzg4NDgsIDMuMzkzMjA1MDQxMDY1ODEyXSwgWzUxLjI0MDI4NjE3NTE2MTYsIDMuMzkzMTYzMzgzMDA3MDQ5Nl0sIFs1MS4yNDAyMTc4NjI2NTA3NSwgMy4zOTMxMTY2OTU4MDYzODRdLCBbNTEuMjQwMTU3ODQ4MjI0MDQ0LCAzLjM5MzA5MzM5NDExNTU2N10sIFs1MS4yNDAwOTc4MzM3OTczMzYsIDMuMzkzMTE4MzcyMTg3MDE4NF0sIFs1MS4yNDAwMzk0OTU3NTEyNiwgMy4zOTMxNTUwMDExMDM4NzhdLCBbNTEuMjM5OTc2MjEyMzgyMzIsIDMuMzkzMTgwMDYyOTk0MzYxXSwgWzUxLjIzOTkxMTE2ODgxMzcwNSwgMy4zOTMxODgzNjEwNzg1MDA3XSwgWzUxLjIzOTg0OTU2MTgyNTM5NSwgMy4zOTMxNjg0MTIxNDg5NTI1XSwgWzUxLjIzOTc5OTUyMTg2MzQ2LCAzLjM5MzEyMzQwMTMyODkyMTNdLCBbNTEuMjM5NzU2MTg3NDI0MDY0LCAzLjM5MzA5Njc0Njg3NjgzNl0sIFs1MS4yMzk3MDc4MjM4NDI3NjQsIDMuMzkzMDg4MzY0OTczNjY0M10sIFs1MS4yMzk2NTQ1MTQ5Mzg1OSwgMy4zOTMwODAwNjY4ODk1MjQ1XSwgWzUxLjIzOTU5NjE3Njg5MjUyLCAzLjM5MzA2MzM4NjkwMjIxM10sIFs1MS4yMzk1NDQ1NDQzNjg5OCwgMy4zOTMwMzUwNTYwNjk0OTMzXSwgWzUxLjIzOTU0MTE5MTYwNzcxNCwgMy4zOTI5NjMzOTA3OTczNzY2XSwgWzUxLjIzOTUzMTIxNzE0Mjk0LCAzLjM5MjkwMzM3NjM3MDY2ODRdLCBbNTEuMjM5NTQyODY3OTg4MzUsIDMuMzkyODY1MDcxMDczMTc0NV0sIFs1MS4yMzk1Njk1MjI0NDA0MywgMy4zOTI4NTE3NDM4NDcxMzE3XSwgWzUxLjIzOTY0NDU0MDQ3MzgyLCAzLjM5Mjg3NTA0NTUzNzk0ODZdLCBbNTEuMjM5Njg3ODc0OTEzMjE2LCAzLjM5MjkwNjcyOTEzMTkzN10sIFs1MS4yMzk3MTk1NTg1MDcyMDQsIDMuMzkyOTUwMDYzNTcxMzM0XSwgWzUxLjIzOTc0NDUzNjU3ODY1NSwgMy4zOTMwMDAwMTk3MTQyMzYzXSwgWzUxLjIzOTc2NzgzODI2OTQ3LCAzLjM5MzA1MzQxMjQzNzQzOV0sIFs1MS4yMzk3ODI4NDE4NzYxNSwgMy4zOTMxMDY3MjEzNDE2MV0sIFs1MS4yMzk3OTQ0OTI3MjE1NiwgMy4zOTMxNTAwNTU3ODEwMDddLCBbNTEuMjM5Nzk0NDkyNzIxNTYsIDMuMzkzMjAzMzY0Njg1MTc4XSwgWzUxLjIzOTc5NDQ5MjcyMTU2LCAzLjM5MzIzMTY5NTUxNzg5NzZdLCBbNTEuMjM5NzcxMTkxMDMwNzQsIDMuMzkzMjQzMzQ2MzYzMzA2XSwgWzUxLjIzOTc0Nzg4OTMzOTkyNCwgMy4zOTMyMzUwNDgyNzkxNjZdLCBbNTEuMjM5NzI3ODU2NTkxMzQ0LCAzLjM5MzIwODM5MzgyNzA4MDddLCBbNTEuMjM5NzE2MjA1NzQ1OTM1LCAzLjM5MzE2ODQxMjE0ODk1MjVdLCBbNTEuMjM5NzA2MjMxMjgxMTYsIDMuMzkzMTM1MDUyMTc0MzI5OF0sIFs1MS4yMzk2OTc4NDkzNzc5OSwgMy4zOTMxMTgzNzIxODcwMTg0XSwgWzUxLjIzOTY4MTE2OTM5MDY4LCAzLjM5MzExNTAxOTQyNTc1XSwgWzUxLjIzOTY0OTU2OTYxNTcyLCAzLjM5MzE3NTAzMzg1MjQ1OF0sIFs1MS4yMzk2MjQ1MDc3MjUyNCwgMy4zOTMyNDUwMjI3NDM5NDA0XSwgWzUxLjIzOTYwNjIzNTE3NjMyNSwgMy4zOTMzMTE3NDI2OTMxODZdLCBbNTEuMjM5NTcyODc1MjAxNywgMy4zOTMzNzY3MDI0NDI3NjUyXSwgWzUxLjIzOTUzMTIxNzE0Mjk0LCAzLjM5MzQzODM5MzI1MDEwNzhdLCBbNTEuMjM5NDgxMTc3MTgxMDA1LCAzLjM5MzQ4MDA1MTMwODg3MDNdLCBbNTEuMjM5NDIyODM5MTM0OTMsIDMuMzkzNTA4MzgyMTQxNTldLCBbNTEuMjM5MzU0NTI2NjI0MDgsIDMuMzkzNTI4NDE0ODkwMTddLCBbNTEuMjM5MjgxMTg0OTcxMzMsIDMuMzkzNTQ2Njg3NDM5MDg0XSwgWzUxLjIzOTIwOTUxOTY5OTIxNiwgMy4zOTM1NjAwMTQ2NjUxMjddLCBbNTEuMjM5MTM0NTAxNjY1ODMsIDMuMzkzNTc4MzcxMDMzMDcyNV0sIFs1MS4yMzkwNTc4OTEwNzA4NCwgMy4zOTM1OTY3Mjc0MDEwMThdLCBbNTEuMjM4OTg0NTQ5NDE4MDksIDMuMzkzNjI2NzM0NjE0MzcyM10sIFs1MS4yMzg5MDk1MzEzODQ3MSwgMy4zOTM2NjUwMzk5MTE4NjZdLCBbNTEuMjM4ODMyODM2OTcwNjksIDMuMzkzNzA1MDIxNTg5OTk0NF0sIFs1MS4yMzg3NTc5MDI3NTYzMywgMy4zOTM3MzY3MDUxODM5ODNdLCBbNTEuMjM4Njc3ODU1NTgxMDQ1LCAzLjM5Mzc2MzM1OTYzNjA2ODNdLCBbNTEuMjM4NTk2MjE1ODQ0MTU0LCAzLjM5Mzc4Njc0NTE0NTkxN10sIFs1MS4yMzg1MTQ0OTIyODgyMywgMy4zOTM4MDY2OTQwNzU0NjVdLCBbNTEuMjM4NDMxMTc2MTcwNzEsIDMuMzkzODI4NDAzMjA0Njc5NV0sIFs1MS4yMzgzNDk1MzY0MzM4MTYsIDMuMzkzODQ0OTk5MzcyOTU5XSwgWzUxLjIzODI3MTE2NTYzOTE2LCAzLjM5Mzg2MDAwMjk3OTYzNl0sIFs1MS4yMzgxOTEyMDIyODI5MDYsIDMuMzkzODc2NjgyOTY2OTQ3Nl0sIFs1MS4yMzgxMTQ1MDc4Njg4ODYsIDMuMzkzOTAwMDY4NDc2Nzk2XSwgWzUxLjIzODA0MTE2NjIxNjEzNSwgMy4zOTM5MzY2OTczOTM2NTU4XSwgWzUxLjIzNzk2NzgyNDU2MzM4NCwgMy4zOTM5OTE2ODI2Nzg0NjFdLCBbNTEuMjM3OTExMTYyODk3OTQ0LCAzLjM5NDA1NjcyNjI0NzA3Ml0sIFs1MS4yMzc4NTc4NTM5OTM3NywgMy4zOTQxNjY2OTY4MTY2ODNdLCBbNTEuMjM3ODIxMjI1MDc2OTE0LCAzLjM5NDI2ODM2OTMwMjE1MzZdLCBbNTEuMjM3NzkyODk0MjQ0MTk0LCAzLjM5NDM4ODM5ODE1NTU3XSwgWzUxLjIzNzc5NjE2MzE4NjQzLCAzLjM5NDUyMDA3Nzg1NDM5NV0sIFs1MS4yMzc4MjI5MDE0NTc1NSwgMy4zOTQ2NzUwNTkyNDQwMzY3XSwgWzUxLjIzNzgwNjIyMTQ3MDI0LCAzLjM5NDgwNTA2MjU2MjIyNzJdXSwKICAgICAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJicm93biIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogZmFsc2UsCiAgImZpbGxDb2xvciI6ICJicm93biIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAibm9DbGlwIjogZmFsc2UsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInNtb290aEZhY3RvciI6IDEuMCwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfODI0MTJhZjU0YjU3NGUxM2JiZTQzMjAxZGYxZjE3MzUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICAgICAgCgogICAgICAgICAgICAgICAgbWFwXzgyNDEyYWY1NGI1NzRlMTNiYmU0MzIwMWRmMWYxNzM1LmZpdEJvdW5kcygKICAgICAgICAgICAgICAgICAgICBbWzUxLjIzNjY4Mjg3ODgwNzE5LCAzLjM5MjYyMTc0NDQyNDEwNDddLCBbNTEuMjQwNjc0NTA4NzM1NTQsIDMuMzk1Mjc2NzEyMjUzNjg5OF1dLAogICAgICAgICAgICAgICAgICAgIHt9CiAgICAgICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgCjwvc2NyaXB0Pg==" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
#@title Functions to create a `timetable`: compare two laps giving a elapsed time per point on the parcours
import math

def midpoint(coords):
    x = 0.0
    y = 0.0
    z = 0.0
    
    for lat, lon in coords:
        latitude = math.radians(lat)
        longitude = math.radians(lon)

        x += math.cos(latitude) * math.cos(longitude)
        y += math.cos(latitude) * math.sin(longitude)
        z += math.sin(latitude)

    total = len(coords)

    x = x / total
    y = y / total
    z = z / total

    central_longitude = math.atan2(y, x)
    central_square_root = math.sqrt(x * x + y * y)
    central_latitude = math.atan2(z, central_square_root)

    return [math.degrees(central_latitude), math.degrees(central_longitude)]
    
def get_timetable(lap1, lap2):
    d, path = compare(lap1, lap2)

    previous_first = path[0][0]
    previous_second = path[0][1]
    timetable = []

    for i in range(0, max(path[-1][0], path[-1][1])+1):
        timetable.append(['', '', '', None, None])

    elapsed = 0
    for first, second in path:
        timetable[int(first)][3]=copy.deepcopy(elapsed)+0
        timetable[int(second)][4]=copy.deepcopy(elapsed)+0
        elapsed+=1

    for i in range(0, len(timetable)):
        point_index_1 = timetable[i][3]
        point_index_2 = timetable[i][4]
        if point_index_1:
            pt = lap1[path[point_index_1][0]]
        elif point_index_2:
            pt = lap2[path[point_index_2][1]]
        avg_pt = midpoint([pt])

        timetable[i][0]=avg_pt[0]
        timetable[i][1]=avg_pt[1]

        if i>0:
            previous_pt = (timetable[i-1][0], timetable[i-1][1])
            timetable[i][2]=timetable[i-1][2]+distance(previous_pt, avg_pt)
        else:
            timetable[i][2]=0

    timetable_df = pd.DataFrame(data=timetable, columns=['latitude','longitude','cumm_distance','elapsed_1', 'elapsed_2'])

    return timetable_df, timetable

def plot_lap(timetable_):
    m = folium.Map(
        width='100%', 
        location=[45.33, -121.69],
        zoom_start=12,
        tiles='openstreetmap'
    )
    lapcoords = get_polyline(np.array([[row[0], row[1]] for row in timetable_]), colour='orange')
    lap = lapcoords.add_to(m)
    m.fit_bounds(lap.get_bounds())

    return m

# Use the functions from the previous cell to create a comparison between two laps.
lap2 = np.array(workout_df.iloc[laps[1][0]:laps[1][1]][['latitude','longitude']].values)
lap6 = np.array(workout_df.iloc[laps[5][0]:laps[5][1]][['latitude','longitude']].values)

timetable_df, timetable_ = get_timetable(lap2, lap6)
plot_lap(timetable_)
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVM9ZmFsc2U7IExfTk9fVE9VQ0g9ZmFsc2U7IExfRElTQUJMRV8zRD1mYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS40LjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NvZGUuanF1ZXJ5LmNvbS9qcXVlcnktMS4xMi40Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS40LjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdjZG4uZ2l0aGFjay5jb20vcHl0aG9uLXZpc3VhbGl6YXRpb24vZm9saXVtL21hc3Rlci9mb2xpdW0vdGVtcGxhdGVzL2xlYWZsZXQuYXdlc29tZS5yb3RhdGUuY3NzIi8+CiAgICA8c3R5bGU+aHRtbCwgYm9keSB7d2lkdGg6IDEwMCU7aGVpZ2h0OiAxMDAlO21hcmdpbjogMDtwYWRkaW5nOiAwO308L3N0eWxlPgogICAgPHN0eWxlPiNtYXAge3Bvc2l0aW9uOmFic29sdXRlO3RvcDowO2JvdHRvbTowO3JpZ2h0OjA7bGVmdDowO308L3N0eWxlPgogICAgCiAgICA8bWV0YSBuYW1lPSJ2aWV3cG9ydCIgY29udGVudD0id2lkdGg9ZGV2aWNlLXdpZHRoLAogICAgICAgIGluaXRpYWwtc2NhbGU9MS4wLCBtYXhpbXVtLXNjYWxlPTEuMCwgdXNlci1zY2FsYWJsZT1ubyIgLz4KICAgIDxzdHlsZT4jbWFwX2I0OWU1MzMzYzgzNTRiY2JhODk3ZmJiZDU3YjhlZmFhIHsKICAgICAgICBwb3NpdGlvbjogcmVsYXRpdmU7CiAgICAgICAgd2lkdGg6IDEwMC4wJTsKICAgICAgICBoZWlnaHQ6IDEwMC4wJTsKICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgIHRvcDogMC4wJTsKICAgICAgICB9CiAgICA8L3N0eWxlPgo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgPGRpdiBjbGFzcz0iZm9saXVtLW1hcCIgaWQ9Im1hcF9iNDllNTMzM2M4MzU0YmNiYTg5N2ZiYmQ1N2I4ZWZhYSIgPjwvZGl2Pgo8L2JvZHk+CjxzY3JpcHQ+ICAgIAogICAgCiAgICAKICAgICAgICB2YXIgYm91bmRzID0gbnVsbDsKICAgIAoKICAgIHZhciBtYXBfYjQ5ZTUzMzNjODM1NGJjYmE4OTdmYmJkNTdiOGVmYWEgPSBMLm1hcCgKICAgICAgICAnbWFwX2I0OWU1MzMzYzgzNTRiY2JhODk3ZmJiZDU3YjhlZmFhJywgewogICAgICAgIGNlbnRlcjogWzQ1LjMzLCAtMTIxLjY5XSwKICAgICAgICB6b29tOiAxMiwKICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICBsYXllcnM6IFtdLAogICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcsCiAgICAgICAgem9vbUNvbnRyb2w6IHRydWUsCiAgICAgICAgfSk7CgoKICAgIAogICAgdmFyIHRpbGVfbGF5ZXJfNDliZGEzNmRlOTE3NDU0OTlkYjJmNGE1MjVkYjc3ZWMgPSBMLnRpbGVMYXllcigKICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgIHsKICAgICAgICAiYXR0cmlidXRpb24iOiBudWxsLAogICAgICAgICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAgICAgICAibWF4TmF0aXZlWm9vbSI6IDE4LAogICAgICAgICJtYXhab29tIjogMTgsCiAgICAgICAgIm1pblpvb20iOiAwLAogICAgICAgICJub1dyYXAiOiBmYWxzZSwKICAgICAgICAib3BhY2l0eSI6IDEsCiAgICAgICAgInN1YmRvbWFpbnMiOiAiYWJjIiwKICAgICAgICAidG1zIjogZmFsc2UKfSkuYWRkVG8obWFwX2I0OWU1MzMzYzgzNTRiY2JhODk3ZmJiZDU3YjhlZmFhKTsKICAgIAogICAgICAgICAgICAgICAgdmFyIHBvbHlfbGluZV8wOGU3MzdhOTJjODE0NzYyODNhMTExNGM4NGEwZjFjZCA9IEwucG9seWxpbmUoCiAgICAgICAgICAgICAgICAgICAgW1s1MS4yMzc3Mjc4NTA2NzU1OCwgMy4zOTQ5NTE3NDU4Njc3Mjk2XSwgWzUxLjIzNzY0Nzg4NzMxOTMzLCAzLjM5NDk3NTA0NzU1ODU0Nl0sIFs1MS4yMzc1NTc4NjU2NzkyNjQsIDMuMzk0OTkwMDUxMTY1MjIzXSwgWzUxLjIzNzQ2OTUyMDQxOTgzNiwgMy4zOTUwMTE2NzY0NzU0MDZdLCBbNTEuMjM3Mzc3OTA2MjE4MTgsIDMuMzk1MDI1MDAzNzAxNDQ5XSwgWzUxLjIzNzI4NDUzMTgxNjg0LCAzLjM5NTAzNTA2MTk4NTI1NDNdLCBbNTEuMjM3MTkyODMzNzk2MTQsIDMuMzk1MDM4NDE0NzQ2NTIzXSwgWzUxLjIzNzEwNDU3MjM1NTc1NCwgMy4zOTUwMzg0MTQ3NDY1MjMzXSwgWzUxLjIzNzAxNzkwMzQ3Njk2LCAzLjM5NTA0MTY4MzY4ODc2MDJdLCBbNTEuMjM2OTM3ODU2MzAxNjY1LCAzLjM5NTA0ODM4OTIxMTI5N10sIFs1MS4yMzY4NTk1NjkzMjYwNSwgMy4zOTUwNTAwNjU1OTE5MzEzXSwgWzUxLjIzNjc4OTQ5NjYxNTUzNiwgMy4zOTUwNjAwNDAwNTY3MDVdLCBbNTEuMjM2NzI0NTM2ODY1OTUsIDMuMzk1MDc4Mzk2NDI0NjUxXSwgWzUxLjIzNjY3MjkwNDM0MjQxLCAzLjM5NTEwODQwMzYzODAwNDRdLCBbNTEuMjM2Njk3ODgyNDEzODY0LCAzLjM5NTE2MTcxMjU0MjE3Nl0sIFs1MS4yMzY2ODc4MjQxMzAwNiwgMy4zOTUyMDY3MjMzNjIyMDddLCBbNTEuMjM2NjgxMjAyNDI2NTUsIDMuMzk1MjM5OTk5NTE3Nzk5XSwgWzUxLjIzNjcwMjgyNzczNjczNSwgMy4zOTUyNzMzNTk0OTI0MjA3XSwgWzUxLjIzNjc5MTE3Mjk5NjE1NiwgMy4zOTUyNjAwMzIyNjYzNzldLCBbNTEuMjM2ODQxMjEyOTU4MSwgMy4zOTUyNTAwNTc4MDE2MDVdLCBbNTEuMjM2OTAxMjI3Mzg0ODA2LCAzLjM5NTIzMDAyNTA1MzAyNDddLCBbNTEuMjM2OTYyODM0MzczMTIsIDMuMzk1MjA4Mzk5NzQyODQxM10sIFs1MS4yMzcwMjk1NTQzMjIzNiwgMy4zOTUxODY2OTA2MTM2Mjc0XSwgWzUxLjIzNzA5NDUxNDA3MTk0LCAzLjM5NTE3MzM2MzM4NzU4NV0sIFs1MS4yMzcxNTc4ODEyNTk5MjUsIDMuMzk1MTYxNzEyNTQyMTc2Ml0sIFs1MS4yMzcyMjExNjQ2Mjg4NiwgMy4zOTUxNDMzNTYxNzQyMzA2XSwgWzUxLjIzNzI4Mjg1NTQzNjIwNiwgMy4zOTUxMjgzNTI1Njc1NTI2XSwgWzUxLjIzNzM0NDU0NjI0MzU1LCAzLjM5NTExODM3ODEwMjc4XSwgWzUxLjIzNzQxOTU2NDI3NjkzNCwgMy4zOTUxMDAwMjE3MzQ4MzRdLCBbNTEuMjM3NDk3ODUxMjUyNTU2LCAzLjM5NTA3ODM5NjQyNDY1MTZdLCBbNTEuMjM3NTc2MjIyMDQ3MjEsIDMuMzk1MDU2Njg3Mjk1NDM3XSwgWzUxLjIzNzY1OTUzODE2NDczNSwgMy4zOTUwMzg0MTQ3NDY1MjNdLCBbNTEuMjM3NzM2MjMyNTc4NzU0LCAzLjM5NTAyMTczNDc1OTIxMl0sIFs1MS4yMzc4MDk0OTA0MTI0NzQsIDMuMzk1MDAwMDI1NjI5OTk3XSwgWzUxLjIzNzg4Nzg2MTIwNzEzLCAzLjM5NDk3MDAxODQxNjY0MzZdLCBbNTEuMjM3OTY5NTAwOTQ0MDIsIDMuMzk0OTMwMDM2NzM4NTE1XSwgWzUxLjIzODA0OTU0ODExOTMxLCAzLjM5NDg5MTczMTQ0MTAyMTRdLCBbNTEuMjM4MTE2MTg0MjQ5NTIsIDMuMzk0ODYzNDAwNjA4MzAxXSwgWzUxLjIzODE4NDQ5Njc2MDM3LCAzLjM5NDgyNTAxMTQ5MTc3NV0sIFs1MS4yMzgyNDI4MzQ4MDY0MzUsIDMuMzk0NzgwMDAwNjcxNzQ0XSwgWzUxLjIzODI4Nzg0NTYyNjQ3LCAzLjM5NDcxMTY4ODE2MDg5NjNdLCBbNTEuMjM4MzIyODgxOTgxNzMsIDMuMzk0NjMxNzI0ODA0NjM5NF0sIFs1MS4yMzgzNDQ1MDcyOTE5MSwgMy4zOTQ1NTMzNTQwMDk5ODY0XSwgWzUxLjIzODM0Nzg2MDA1MzE4LCAzLjM5NDQ2ODM2MTUxMTgyNl0sIFs1MS4yMzgzNDk1MzY0MzM4MTYsIDMuMzk0Mzg1MDQ1Mzk0MzAyXSwgWzUxLjIzODM1MTIxMjgxNDQ2LCAzLjM5NDMwMzQwNTY1NzQxMV0sIFs1MS4yMzgzNjc4OTI4MDE3NiwgMy4zOTQyNDMzOTEyMzA3MDNdLCBbNTEuMjM4NDIyODc4MDg2NTcsIDMuMzk0MjIwMDA1NzIwODUzNF0sIFs1MS4yMzg0NTQ1NjE2ODA1NTUsIDMuMzk0MjI4Mzg3NjI0MDI1XSwgWzUxLjIzODQ4NjE2MTQ1NTUxLCAzLjM5NDI3MTcyMjA2MzQyMl0sIFs1MS4yMzg1MTEyMjMzNDU5OTUsIDMuMzk0MzUwMDA5MDM5MDQ0NF0sIFs1MS4yMzg1MjI4NzQxOTE0LCAzLjM5NDQ0NTA1OTgyMTAwOTZdLCBbNTEuMjM4NTMxMTcyMjc1NTQsIDMuMzk0NTQ4NDA4Njg3MTE0N10sIFs1MS4yMzg1Mzk1NTQxNzg3MTUsIDMuMzk0NjU1MDI2NDk1NDU2M10sIFs1MS4yMzg1NDYxNzU4ODIyMiwgMy4zOTQ3NTgzNzUzNjE1NjEzXSwgWzUxLjIzODU0OTUyODY0MzQ5NiwgMy4zOTQ4NDY3MjA2MjA5OTAyXSwgWzUxLjIzODU1NDU1Nzc4NTM5LCAzLjM5NDkxMDAwMzk4OTkzNDVdLCBbNTEuMjM4NTgxMjEyMjM3NDgsIDMuMzk0OTE2NzA5NTEyNDcyXSwgWzUxLjIzODU2OTU2MTM5MjA3NiwgMy4zOTQ4OTY2NzY3NjM4OTI2XSwgWzUxLjIzODU4OTUxMDMyMTYyLCAzLjM5NDgzMTcxNzAxNDMxM10sIFs1MS4yMzg1NjYyMDg2MzA4LCAzLjM5NDc2MDA1MTc0MjE5Nl0sIFs1MS4yMzg1NTI4ODE0MDQ3NiwgMy4zOTQ2NjUwMDA5NjAyMzEzXSwgWzUxLjIzODUzNjIwMTQxNzQ0NiwgMy4zOTQ1NjY2ODEyMzYwMjg3XSwgWzUxLjIzODUxOTUyMTQzMDEzLCAzLjM5NDQ2MDA2MzQyNzY4NjJdLCBbNTEuMjM4NTA3ODcwNTg0NzI2LCAzLjM5NDM1MTY4NTQxOTY3OV0sIFs1MS4yMzg0OTYyMTk3MzkzMiwgMy4zOTQyNDgzMzY1NTM1NzNdLCBbNTEuMjM4NDg0NTY4ODkzOTEsIDMuMzk0MTU1MDQ1OTcxMjc0NF0sIFs1MS4yMzg0NzQ1MTA2MTAxMDQsIDMuMzk0MDcxNzI5ODUzNzQ5M10sIFs1MS4yMzg0OTQ1NDMzNTg2ODQsIDMuMzk0MDM4MzY5ODc5MTI2NV0sIFs1MS4yMzg0OTExOTA1OTc0MTUsIDMuMzk0MDM2NjkzNDk4NDkyMl0sIFs1MS4yMzg0OTExOTA1OTc0MTUsIDMuMzk0MDM2NjkzNDk4NDkyMl0sIFs1MS4yMzg1Mjk0OTU4OTQ5LCAzLjM5NDEwNTAwNjAwOTM0XSwgWzUxLjIzODUzMjg0ODY1NjE4NSwgMy4zOTQxOTY3MDQwMzAwMzY1XSwgWzUxLjIzODUzOTU1NDE3ODcxNSwgMy4zOTQyOTMzNDczNzM2MDQ4XSwgWzUxLjIzODU1NzgyNjcyNzYyLCAzLjM5NDQxMDAyMzQ2NTc1M10sIFs1MS4yMzg1NzEyMzc3NzI3MSwgMy4zOTQ1MjUwMjMxNzcyNjddLCBbNTEuMjM4NTgxMjEyMjM3NDgsIDMuMzk0NjM2NjcwMTI3NTExNV0sIFs1MS4yMzg1ODQ1NjQ5OTg3NDYsIDMuMzk0NzMwMDQ0NTI4ODQyNF0sIFs1MS4yMzg1OTExODY3MDIyNSwgMy4zOTQ4MDg0MTUzMjM0OTZdLCBbNTEuMjM4NTg0NTY0OTk4NzQ2LCAzLjM5NDg3NjcyNzgzNDM0MzVdLCBbNTEuMjM4NTk0NTM5NDYzNTIsIDMuMzk0OTEwMDAzOTg5OTM0NV0sIFs1MS4yMzg2MTEyMTk0NTA4MywgMy4zOTQ5MTAwMDM5ODk5MzVdLCBbNTEuMjM4NjE3ODQxMTU0MzQ0LCAzLjM5NDg0NjcyMDYyMDk5MDJdLCBbNTEuMjM4NjA2MTkwMzA4OTMsIDMuMzk0Nzg4MzgyNTc0OTE2XSwgWzUxLjIzODU5MTE4NjcwMjI1LCAzLjM5NDcxMDAxMTc4MDI2MV0sIFs1MS4yMzg1Nzc4NTk0NzYyMSwgMy4zOTQ2MjY2OTU2NjI3MzddLCBbNTEuMjM4NTc5NTM1ODU2ODUsIDMuMzk0NTY4MzU3NjE2NjYzNF0sIFs1MS4yMzg2MDQ1MTM5MjgyOSwgMy4zOTQ1MDAwNDUxMDU4MTU0XSwgWzUxLjIzODY0MTIyNjY2NDE4NiwgMy4zOTQ0MjUwMjcwNzI0M10sIFs1MS4yMzg2OTk1NjQ3MTAyNiwgMy4zOTQzMjMzNTQ1ODY5NTldLCBbNTEuMjM4NzM5NTQ2Mzg4MzksIDMuMzk0MjI4Mzg3NjI0MDI1M10sIFs1MS4yMzg3NzI5MDYzNjMwMSwgMy4zOTQxMzMzMzY4NDIwNl0sIFs1MS4yMzg3OTk1NjA4MTUwOTYsIDMuMzk0MDQxNzIyNjQwMzk2XSwgWzUxLjIzODgyMTE4NjEyNTI4LCAzLjM5Mzk1NjczMDE0MjIzNl0sIFs1MS4yMzg4NDc4NDA1NzczNjQsIDMuMzkzODg1MDY0ODcwMTE5XSwgWzUxLjIzODg3NjE3MTQxMDA4NCwgMy4zOTM4MzMzNDg1Mjc1NTA3XSwgWzUxLjIzODkwNDUwMjI0MjgwNCwgMy4zOTM4MDUwMTc2OTQ4MzFdLCBbNTEuMjM4OTM3ODYyMjE3NDI2LCAzLjM5Mzc4MTcxNjAwNDAxNDVdLCBbNTEuMjM4OTY5NTQ1ODExNDE1LCAzLjM5Mzc3MzMzNDEwMDg0MjVdLCBbNTEuMjM4OTk0NTIzODgyODYsIDMuMzkzNzc4MzYzMjQyNzQ1XSwgWzUxLjIzODk3NDQ5MTEzNDI5LCAzLjM5Mzg0MDA1NDA1MDA4NzVdLCBbNTEuMjM4OTAyODI1ODYyMTcsIDMuMzkzOTY2NzA0NjA3MDEwM10sIFs1MS4yMzg4NzQ0OTUwMjk0NSwgMy4zOTQwNjE2NzE1Njk5NDNdLCBbNTEuMjM4ODQxMjE4ODczODY2LCAzLjM5NDE1NTA0NTk3MTI3NF0sIFs1MS4yMzg3OTc4ODQ0MzQ0NiwgMy4zOTQyNTgzOTQ4MzczOF0sIFs1MS4yMzg3NTk0OTUzMTc5MzYsIDMuMzk0MzUwMDA5MDM5MDQ0NF0sIFs1MS4yMzg3MTYxNjA4Nzg1NCwgMy4zOTQ0NDE3MDcwNTk3NDFdLCBbNTEuMjM4Njc5NTMxOTYxNjcsIDMuMzk0NTM4MzUwNDAzMzA5M10sIFs1MS4yMzg2NDk1MjQ3NDgzMiwgMy4zOTQ2NTAwODExNzI1ODVdLCBbNTEuMjM4NjMyODQ0NzYxMDE0LCAzLjM5NDc2NTA4MDg4NDA5ODZdLCBbNTEuMjM4NjMxMTY4MzgwMzgsIDMuMzk0ODgzMzQ5NTM3ODQ5NF0sIFs1MS4yMzg2NzEyMzM4Nzc1NSwgMy4zOTQ5NDgzOTMxMDY0NjE1XSwgWzUxLjIzODcyMjg2NjQwMTA3NiwgMy4zOTUwMDAwMjU2Mjk5OTczXSwgWzUxLjIzODc2NjIwMDg0MDQ3LCAzLjM5NTA1NTAxMDkxNDgwM10sIFs1MS4yMzg4MjI4NjI1MDU5MSwgMy4zOTUwNjY3NDU1NzkyNDI3XSwgWzUxLjIzODg2Mjg0NDE4NDA1LCAzLjM5NTA0MzM2MDA2OTM5NDZdLCBbNTEuMjM4ODc5NTI0MTcxMzUsIDMuMzk1MDEwMDAwMDk0NzcxNF0sIFs1MS4yMzg4Nzk1MjQxNzEzNSwgMy4zOTQ5NjUwNzMwOTM3NzJdLCBbNTEuMjM4ODgyODc2OTMyNjIsIDMuMzk0OTAzMzgyMjg2NDI5NF0sIFs1MS4yMzg4NTc4OTg4NjExNywgMy4zOTQ4NTE2NjU5NDM4NjFdLCBbNTEuMjM4ODM2MTg5NzMxOTYsIDMuMzk0Nzg2NzA2MTk0MjgxNl0sIFs1MS4yMzg4MTk1MDk3NDQ2NDQsIDMuMzk0NzEzMzY0NTQxNTMwNl0sIFs1MS4yMzg4MzYxODk3MzE5NiwgMy4zOTQ2NTgzNzkyNTY3MjU4XSwgWzUxLjIzODg2Mjg0NDE4NDA0LCAzLjM5NDYzMTcyNDgwNDYzOTRdLCBbNTEuMjM4OTAxMjMzMzAwNTcsIDMuMzk0NjIwMDczOTU5MjMyXSwgWzUxLjIzODkzNDUwOTQ1NjE2LCAzLjM5NDU4NTAzNzYwMzk3NF0sIFs1MS4yMzg5NTYyMTg1ODUzNywgMy4zOTQ1MjAwNzc4NTQzOTU0XSwgWzUxLjIzODk1OTU3MTM0NjY1LCAzLjM5NDQ1MTY4MTUyNDUxNTZdLCBbNTEuMjM4OTEyODg0MTQ1OTcsIDMuMzk0Mzk2Njk2MjM5NzFdLCBbNTEuMjM4ODg0NTUzMzEzMjYsIDMuMzk0MzM1MDA1NDMyMzY3OF0sIFs1MS4yMzg4NTc4OTg4NjExNywgMy4zOTQyNjMzNDAxNjAyNTA3XSwgWzUxLjIzODg2Nzg3MzMyNTk0NCwgMy4zOTQxODMzNzY4MDM5OTRdLCBbNTEuMjM4ODg0NTUzMzEzMjYsIDMuMzk0MTMwMDY3ODk5ODIzNl0sIFs1MS4yMzg5MTEyMDc3NjUzNSwgMy4zOTQwOTMzNTUxNjM5MzIzXSwgWzUxLjIzODkyNzg4Nzc1MjY1LCAzLjM5NDEwNjY4MjM4OTk3MzddLCBbNTEuMjM4OTUyODY1ODI0MSwgMy4zOTQxMzMzMzY4NDIwNTk2XSwgWzUxLjIzODk3OTUyMDI3NjE5NiwgMy4zOTQxNjUwMjA0MzYwNDldLCBbNTEuMjM5MDAxMjI5NDA1NDEsIDMuMzk0MzAwMDUyODk2MTQzXSwgWzUxLjIzOTAxMjg4MDI1MDgxLCAzLjM5NDQwMDA0OTAwMDk3ODVdLCBbNTEuMjM4OTk5NTUzMDI0NzcsIDMuMzk0NTA4MzQzMTg5OTU0M10sIFs1MS4yMzg5ODI4NzMwMzc0NjUsIDMuMzk0NjIwMDczOTU5MjMyXSwgWzUxLjIzODk4MTE5NjY1NjgyLCAzLjM5NDcyMzMzOTAwNjMwNDddLCBbNTEuMjM4OTg0NTQ5NDE4MSwgMy4zOTQ4MjAwNjYxNjg5MDQzXSwgWzUxLjIzODk3NDQ5MTEzNDI5LCAzLjM5NDg4NTAyNTkxODQ4MzddLCBbNTEuMjM5MDExMjAzODcwMTg0LCAzLjM5NDg5NjY3Njc2Mzg5MjZdLCBbNTEuMjM5MDQ2MjQwMjI1NDM0LCAzLjM5NDg2NTA3Njk4ODkzNl0sIFs1MS4yMzkwMzI4MjkxODAzNiwgMy4zOTQ4NTE2NjU5NDM4NjFdLCBbNTEuMjM5MDIyODU0NzE1NTg2LCAzLjM5NDc4ODM4MjU3NDkxNTRdLCBbNTEuMjM5MDA5NTI3NDg5NTQsIDMuMzk0NzAwMDM3MzE1NDg4XSwgWzUxLjIzOTAwNzg1MTEwODkxLCAzLjM5NDU5NTAxMjA2ODc0ODVdLCBbNTEuMjM5MDE0NTU2NjMxNDQ2LCAzLjM5NDQ5MzMzOTU4MzI3NzddLCBbNTEuMjM5MDIyODU0NzE1NTg2LCAzLjM5NDM4MzM2OTAxMzY2N10sIFs1MS4yMzkwMjc4ODM4NTc0OSwgMy4zOTQyODg0MDIwNTA3MzQ1XSwgWzUxLjIzOTA0NjI0MDIyNTQzNCwgMy4zOTQxOTMzNTEyNjg3Njg4XSwgWzUxLjIzOTA3NDU3MTA1ODE1LCAzLjM5NDExNjc0MDY3Mzc4MDRdLCBbNTEuMjM5MTQxMjA3MTg4MzcsIDMuMzk0MTAwMDYwNjg2NDY5XSwgWzUxLjIzOTIwMTIyMTYxNTA4LCAzLjM5NDA5NjcwNzkyNTIwMTRdLCBbNTEuMjM5MjU5NTU5NjYxMTYsIDMuMzk0MTQ4MzQwNDQ4NzM3XSwgWzUxLjIzOTMxMjg2ODU2NTMyLCAzLjM5NDE5MzM1MTI2ODc2ODhdLCBbNTEuMjM5MzQ2MjI4NTM5OTQ0LCAzLjM5NDI0ODMzNjU1MzU3M10sIFs1MS4yMzkzMDQ1NzA0ODExOSwgMy4zOTQzMTg0MDkyNjQwODddLCBbNTEuMjM5MzAxMjE3NzE5OTEsIDMuMzk0Mzk2Njk2MjM5NzEwM10sIFs1MS4yMzkyOTExNTk0MzYxMSwgMy4zOTQ0OTY2OTIzNDQ1NDY4XSwgWzUxLjIzOTI4Mjg2MTM1MTk3LCAzLjM5NDYxMDAxNTY3NTQyNTVdLCBbNTEuMjM5Mjg3ODkwNDkzODgsIDMuMzk0NzEzMzY0NTQxNTMwNl0sIFs1MS4yMzkyOTExNTk0MzYxMSwgMy4zOTQ4MDY3Mzg5NDI4NjJdLCBbNTEuMjM5Mjg0NTM3NzMyNiwgMy4zOTQ4ODMzNDk1Mzc4NDldLCBbNTEuMjM5MzA0NTcwNDgxMTksIDMuMzk0OTA1MDU4NjY3MDYzN10sIFs1MS4yMzkyOTYxODg1NzgwMSwgMy4zOTQ5MTY3MDk1MTI0NzJdLCBbNTEuMjM5MjQxMjAzMjkzMjA0LCAzLjM5NDg4MzM0OTUzNzg0OTRdLCBbNTEuMjM5MjE2MjI1MjIxNzUsIDMuMzk0ODI4MzY0MjUzMDQ0XSwgWzUxLjIzOTIwMjg5Nzk5NTcxLCAzLjM5NDc1MTY2OTgzOTAyNF0sIFs1MS4yMzkxOTk1NDUyMzQ0NCwgMy4zOTQ2NjY2NzczNDA4NjVdLCBbNTEuMjM5MjE2MjI1MjIxNzUsIDMuMzk0NTc1MDYzMTM5MTk5OF0sIFs1MS4yMzkyNDYyMzI0MzUxMSwgMy4zOTQ0ODY3MTc4Nzk3NzI2XSwgWzUxLjIzOTMwMTIxNzcxOTkxLCAzLjM5NDQ0NTA1OTgyMTAxXSwgWzUxLjIzOTM1OTU1NTc2NTk5LCAzLjM5NDQyNTAyNzA3MjQzXSwgWzUxLjIzOTM5Nzg2MTA2MzQ3LCAzLjM5NDQxODQwNTM2ODkyNDZdLCBbNTEuMjM5NDUyODQ2MzQ4Mjg2LCAzLjM5NDQzNTAwMTUzNzIwMzNdLCBbNTEuMjM5NDgxMTc3MTgxMDA1LCAzLjM5NDQ0NjczNjIwMTY0MzVdLCBbNTEuMjM5NDk3ODU3MTY4MzIsIDMuMzk0NDc2NzQzNDE0OTk4XSwgWzUxLjIzOTUyNjE4ODAwMTA0LCAzLjM5NDUzMTcyODY5OTgwM10sIFs1MS4yMzk1MDI4ODYzMTAyMiwgMy4zOTQ3MDAwMzczMTU0ODhdLCBbNTEuMjM5NTA5NTA4MDEzNzI1LCAzLjM5NDgwNjczODk0Mjg2MTZdLCBbNTEuMjM5NTI0NTExNjIwNCwgMy4zOTQ5MTMzNTY3NTEyMDM1XSwgWzUxLjIzOTU4NDUyNjA0NzExLCAzLjM5NDk2MDA0Mzk1MTg2OTVdLCBbNTEuMjM5NjI0NTA3NzI1MjMsIDMuMzk0OTkzNDAzOTI2NDkxM10sIFs1MS4yMzk2NTk1NDQwODA1LCAzLjM5NDk5NTA4MDMwNzEyNjVdLCBbNTEuMjM5NjUxMTYyMTc3MzI0LCAzLjM5NDkzMTcxMzExOTE0OV0sIFs1MS4yMzk2NDExODc3MTI1NiwgMy4zOTQ4Nzg0MDQyMTQ5NzhdLCBbNTEuMjM5NjI3ODYwNDg2NTEsIDMuMzk0ODA4NDE1MzIzNDk2XSwgWzUxLjIzOTYyMjgzMTM0NDYwNCwgMy4zOTQ3MDgzMzUzOTk2Mjc3XSwgWzUxLjIzOTYyMTIzODc4MywgMy4zOTQ2MjMzNDI5MDE0Njg3XSwgWzUxLjIzOTU5NjE3Njg5MjUyNiwgMy4zOTQ1MTUwNDg3MTI0OTI0XSwgWzUxLjIzOTU5NDUwMDUxMTg4LCAzLjM5NDQ1ODM4NzA0NzA1M10sIFs1MS4yMzk1ODYyMDI0Mjc3NSwgMy4zOTQzODY3MjE3NzQ5MzU3XSwgWzUxLjIzOTU3OTQ5NjkwNTIxLCAzLjM5NDMyMzM1NDU4Njk1OV0sIFs1MS4yMzk1Nzc5MDQzNDM2MDUsIDMuMzk0MjY2NjkyOTIxNTE5M10sIFs1MS4yMzk1NzExOTg4MjEwNywgMy4zOTQxODY3Mjk1NjUyNjMyXSwgWzUxLjIzOTU3NjIyNzk2Mjk3LCAzLjM5NDE0Njc0Nzg4NzEzNV0sIFs1MS4yMzk1NzYyMjc5NjI5NjQsIDMuMzk0MDYwMDc5MDA4MzQwNF0sIFs1MS4yMzk1NzQ1NTE1ODIzMzYsIDMuMzkzOTczNDEwMTI5NTQ2N10sIFs1MS4yMzk1Njc4NDYwNTk3OSwgMy4zOTM4ODE3MTIxMDg4NDk2XSwgWzUxLjIzOTU2MjkwMDczNjkzLCAzLjM5Mzc5MTY5MDQ2ODc4ODZdLCBbNTEuMjM5NTYxMjI0MzU2Mjk0LCAzLjM5MzcwNjY5Nzk3MDYyOV0sIFs1MS4yMzk1NzExOTg4MjEwNywgMy4zOTM2MzMzNTYzMTc4Nzc4XSwgWzUxLjIzOTU4NjIwMjQyNzc1LCAzLjM5MzU3MTY2NTUxMDUzNTJdLCBbNTEuMjM5NjA2MjM1MTc2MzI1LCAzLjM5MzUxNjY4MDIyNTczMDRdLCBbNTEuMjM5NjU5NTQ0MDgwNDk2LCAzLjM5MzQ3MzM0NTc4NjMzM10sIFs1MS4yMzk3NDExODM4MTczOTQsIDMuMzkzNDQzMzM4NTcyOTc4NV0sIFs1MS4yMzk4MDk0OTYzMjgyNCwgMy4zOTM0MTUwMDc3NDAyNTk2XSwgWzUxLjIzOTg4Nzg2NzEyMjg4LCAzLjM5MzM5NjczNTE5MTM0NDhdLCBbNTEuMjM5OTY2MjM3OTE3NTQsIDMuMzkzMzkwMDI5NjY4ODA3NV0sIFs1MS4yNDAwNDExNzIxMzE4OTYsIDMuMzkzMzg1MDAwNTI2OTA1NV0sIFs1MS4yNDAxMTc4NjY1NDU5MTYsIDMuMzkzMzgwMDU1MjA0MDM0XSwgWzUxLjI0MDE5NDU2MDk1OTkzNSwgMy4zOTMzNzUwMjYwNjIxMzFdLCBbNTEuMjQwMjY5NDk1MTc0MjksIDMuMzkzMzczMzQ5NjgxNDk2Nl0sIFs1MS4yNDAzNDI4MzY4MjcwNSwgMy4zOTMzNzAwODA3MzkyNTk3XSwgWzUxLjI0MDQxMjgyNTcxODUyLCAzLjM5MzM2NTA1MTU5NzM1N10sIFs1MS4yNDA0NzExNjM3NjQ1OTYsIDMuMzkzMzU2NjY5Njk0MTg1M10sIFs1MS4yNDA1MzExNzgxOTEzMDQsIDMuMzkzMzQzMzQyNDY4MTQyXSwgWzUxLjI0MDU1NDU2MzcwMTE0NiwgMy4zOTMzMDAwMDgwMjg3NDU3XSwgWzUxLjI0MDU4NjE2MzQ3NjExLCAzLjM5MzI1NTA4MTAyNzc0NThdLCBbNTEuMjQwNTk5NDkwNzAyMTUsIDMuMzkzMTkxNzEzODM5NzY5NF0sIFs1MS4yNDA1OTExOTI2MTgwMDUsIDMuMzkzMTUxNzMyMTYxNjQwN10sIFs1MS4yNDA1Nzk1NDE3NzI2MDQsIDMuMzkzMTMxNjk5NDEzMDYxXSwgWzUxLjI0MDU0MTIzNjQ3NTExLCAzLjM5MzEwODM5NzcyMjI0NDddLCBbNTEuMjQwNDcxMTYzNzY0NTk2LCAzLjM5MzA5MDA0MTM1NDI5ODZdLCBbNTEuMjQwNDExMjMzMTU2OTIsIDMuMzkzMDUwMDU5Njc2MTcwM10sIFs1MS4yNDAzNzExNjc2NTk3NiwgMy4zOTI5OTE3MjE2MzAwOTY0XSwgWzUxLjI0MDM1MTIxODczMDIxLCAzLjM5MjkzNjczNjM0NTI5MV0sIFs1MS4yNDAzMjk1MDk2MDA5OSwgMy4zOTI5MDg0MDU1MTI1NzFdLCBbNTEuMjQwMzI0NTY0Mjc4MTI2LCAzLjM5MjkxMDA4MTg5MzIwNV0sIFs1MS4yNDAzNjc4OTg3MTc1MiwgMy4zOTI5MzY3MzYzNDUyOTA3XSwgWzUxLjI0MDQzMjg1ODQ2NzEsIDMuMzkyOTUxNzM5OTUxOTY4XSwgWzUxLjI0MDQzMjg1ODQ2NzEsIDMuMzkyOTI4MzU0NDQyMTE5XSwgWzUxLjI0MDQ0MjgzMjkzMTg3LCAzLjM5Mjg3ODM5ODI5OTIxNzJdLCBbNTEuMjQwNDIyODg0MDAyMzMsIDMuMzkyODE2NzA3NDkxODc1XSwgWzUxLjI0MDQxNzg1NDg2MDQyNSwgMy4zOTI3NDgzOTQ5ODEwMjY2XSwgWzUxLjI0MDM5OTQ5ODQ5MjQ4NiwgMy4zOTI2ODUwMjc3OTMwNTAzXSwgWzUxLjI0MDM2NjIyMjMzNjg5LCAzLjM5MjY1NTAyMDU3OTY5NTddLCBbNTEuMjQwMzA5NTYwNjcxNDU2LCAzLjM5MjY2MzQwMjQ4Mjg2NzJdLCBbNTEuMjQwMjUyODk5MDA2MDEsIDMuMzkyNjgzMzUxNDEyNDE2XSwgWzUxLjI0MDE5NjIzNzM0MDU3LCAzLjM5MjcwODQxMzMwMjg5OV0sIFs1MS4yNDAxMzk0OTE4NTYxLCAzLjM5MjczODMzNjY5NzIyMTJdLCBbNTEuMjQwMDc5NTYxMjQ4NDIsIDMuMzkyNzc1MDQ5NDMzMTEyXSwgWzUxLjI0MDAxNDUxNzY3OTgsIDMuMzkyODA1MDU2NjQ2NDY1NF0sIFs1MS4yMzk5NTEyMzQzMTA4NjUsIDMuMzkyODM2NzQwMjQwNDU0Ml0sIFs1MS4yMzk4ODk1NDM1MDM1MiwgMy4zOTI4NjY3NDc0NTM4MDldLCBbNTEuMjM5ODM5NTAzNTQxNTksIDMuMzkyODkzNDAxOTA1ODk0M10sIFs1MS4yMzk3ODQ1MTgyNTY3OCwgMy4zOTI5MDY3MjkxMzE5MzddLCBbNTEuMjM5NzM2MjM4NDk0NTE1LCAzLjM5MjkyMDA1NjM1Nzk3OTNdLCBbNTEuMjM5NzI3ODU2NTkxMzQ0LCAzLjM5Mjk1MzMzMjUxMzU3MV0sIFs1MS4yMzk3NDQ1MzY1Nzg2NTUsIDMuMzkyOTU2Njg1Mjc0ODRdLCBbNTEuMjM5NzUyODM0NjYyOCwgMy4zOTI5NTgzNjE2NTU0NzRdLCBbNTEuMjM5Nzg0NTE4MjU2NzgsIDMuMzkyOTYxNzE0NDE2NzQyM10sIFs1MS4yMzk4Mjk1MjkwNzY4MiwgMy4zOTI5NDgzODcxOTA2OTk2XSwgWzUxLjIzOTg5NjE2NTIwNzAzNSwgMy4zOTI5MjY2NzgwNjE0ODUzXSwgWzUxLjIzOTk0NDUyODc4ODMzLCAzLjM5MjkwODQwNTUxMjU3MV0sIFs1MS4yNDAwMDQ1NDMyMTUwMywgMy4zOTI4NjgzNDAwMTU0MTE0XSwgWzUxLjI0MDA2NjIzNDAyMjM4NiwgMy4zOTI4NTUwMTI3ODkzNjg2XSwgWzUxLjI0MDEzMjg3MDE1MjU5LCAzLjM5MjgzNjc0MDI0MDQ1NDddLCBbNTEuMjQwMTk5NTA2MjgyODEsIDMuMzkyODA4NDA5NDA3NzM1XSwgWzUxLjI0MDI2NzkwMjYxMjY4NiwgMy4zOTI3ODMzNDc1MTcyNTI0XSwgWzUxLjI0MDMzNzg5MTUwNDE3LCAzLjM5Mjc3MDAyMDI5MTIwOTddLCBbNTEuMjQwMzk2MjI5NTUwMjQsIDMuMzkyNzY4MzQzOTEwNTc0NV0sIFs1MS4yNDA0NDc4NjIwNzM3OCwgMy4zOTI3ODE2NzExMzY2MTc3XSwgWzUxLjI0MDQ5Mjg3Mjg5MzgxLCAzLjM5MjgxNTAzMTExMTI0MDRdLCBbNTEuMjQwNTA5NTUyODgxMTIsIDMuMzkyODU1MDEyNzg5MzY5NV0sIFs1MS4yNDA0NDk1Mzg0NTQ0MSwgMy4zOTI4NTUwMTI3ODkzNjg2XSwgWzUxLjI0MDQzMjg1ODQ2NzEsIDMuMzkyODQxNjg1NTYzMzI1NF0sIFs1MS4yNDA0MjEyMDc2MjE2OTQsIDMuMzkyODAxNzAzODg1MTk3Nl0sIFs1MS4yNDA0MTQ1MDIwOTkxNTYsIDMuMzkyNzcxNjk2NjcxODQzNV0sIFs1MS4yNDAzNDc4NjU5Njg5NSwgMy4zOTI3ODY3MDAyNzg1MjA2XSwgWzUxLjI0MDI5NDU1NzA2NDc3LCAzLjM5MjgxNTAzMTExMTI0MDRdLCBbNTEuMjQwMjM5NTcxNzc5OTcsIDMuMzkyODQ1MDM4MzI0NTk1XSwgWzUxLjI0MDE3NjIwNDU5MiwgMy4zOTI4NzE2OTI3NzY2OF0sIFs1MS4yNDAxMTk1NDI5MjY1NSwgMy4zOTI4OTM0MDE5MDU4OTQzXSwgWzUxLjI0MDA2MTIwNDg4MDQ4LCAzLjM5MjkxMzM1MDgzNTQ0M10sIFs1MS4yMzk5OTYxNjEzMTE4NjUsIDMuMzkyOTI2Njc4MDYxNDg1XSwgWzUxLjIzOTkzNjIzMDcwNDE5NSwgMy4zOTI5MzMzODM1ODQwMjNdLCBbNTEuMjM5ODc3ODkyNjU4MTE0LCAzLjM5Mjk0MTY4MTY2ODE2MjNdLCBbNTEuMjM5ODI2MTc2MzE1NTQ2LCAzLjM5Mjk0NTAzNDQyOTQzMV0sIFs1MS4yMzk3ODQ1MTgyNTY3OCwgMy4zOTI5NDMzNTgwNDg3OTY3XSwgWzUxLjIzOTc2MTIxNjU2NTk3NCwgMy4zOTI5NzUwNDE2NDI3ODVdLCBbNTEuMjM5NzU5NTQwMTg1MzMsIDMuMzkyOTkzMzk4MDEwNzMwN10sIFs1MS4yMzk3NTYxODc0MjQwNjQsIDMuMzkzMDIwMDUyNDYyODE2Ml0sIFs1MS4yMzk3NjYxNjE4ODg4NCwgMy4zOTMwMzAwMjY5Mjc1OTA0XSwgWzUxLjIzOTgyMTIzMDk5MjY3NSwgMy4zOTMwMjE3Mjg4NDM0NV0sIFs1MS4yMzk4NzI4NjM1MTYyMSwgMy4zOTMwMDY3MjUyMzY3NzNdLCBbNTEuMjM5OTMyODc3OTQyOTIsIDMuMzkyOTgwMDcwNzg0Njg4NF0sIFs1MS4yNDAwMjEyMjMyMDIzNCwgMy4zOTI5NDE2ODE2NjgxNjJdLCBbNTEuMjQwMDk3ODMzNzk3MzQsIDMuMzkyOTIzNDA5MTE5MjQ5XSwgWzUxLjI0MDE1OTUyNDYwNDY4LCAzLjM5MjkxODM3OTk3NzM0NV0sIFs1MS4yNDAyMDYyMTE4MDUzNDQsIDMuMzkyOTIwMDU2MzU3OTc5M10sIFs1MS4yNDAyMTc4NjI2NTA3NSwgMy4zOTI5NTMzMzI1MTM1NzFdLCBbNTEuMjQwMjQyODQwNzIyMiwgMy4zOTI5ODUwMTYxMDc1NTk2XSwgWzUxLjI0MDI1Mjg5OTAwNjAxLCAzLjM5MzAxMzM0Njk0MDI3OTVdLCBbNTEuMjQwMjUxMjIyNjI1Mzc1LCAzLjM5MzA0MTY3Nzc3Mjk5OV0sIFs1MS4yNDAyMzQ1NDI2MzgwNiwgMy4zOTMwNjY3Mzk2NjM0ODJdLCBbNTEuMjQwMTg2MTc5MDU2NzY0LCAzLjM5MzA4MTc0MzI3MDE1ODhdLCBbNTEuMjQwMTA3ODkyMDgxMTQsIDMuMzkzMDc4MzkwNTA4ODldLCBbNTEuMjQwMDc3ODg0ODY3NzksIDMuMzkzMTAwMDE1ODE5MDczXSwgWzUxLjI0MDA1NjE3NTczODU4LCAzLjM5MzEzNTA1MjE3NDMyOTNdLCBbNTEuMjQwMDYyODgxMjYxMTIsIDMuMzkzMTYxNzA2NjI2NDE1N10sIFs1MS4yNDAwNjc4MjY1ODM5OCwgMy4zOTMxODMzMzE5MzY1OThdLCBbNTEuMjQwMDcyODU1NzI1ODg0LCAzLjM5MzE5Njc0Mjk4MTY3MjNdLCBbNTEuMjQwMDk2MTU3NDE2NywgMy4zOTMxODY2ODQ2OTc4NjY0XSwgWzUxLjI0MDEzOTQ5MTg1NjEsIDMuMzkzMTYwMDMwMjQ1NzgxXSwgWzUxLjI0MDIwMjg1OTA0NDA3NSwgMy4zOTMxMzE2OTk0MTMwNjE2XSwgWzUxLjI0MDI2OTQ5NTE3NDI5NiwgMy4zOTMxNDY3MDMwMTk3Mzg2XSwgWzUxLjI0MDMyOTUwOTYwMDk5LCAzLjM5MzE2NTA1OTM4NzY4MzRdLCBbNTEuMjQwMzg0NDk0ODg1Nzk1LCAzLjM5MzE4MzMzMTkzNjU5NzRdLCBbNTEuMjQwNDMxMTgyMDg2NDcsIDMuMzkzMjA2NzE3NDQ2NDQ3XSwgWzUxLjI0MDQ0NjE4NTY5MzE0NSwgMy4zOTMyNTY2NzM1ODkzNDkyXSwgWzUxLjI0MDQ2Nzg5NDgyMjM2LCAzLjM5MzMwMzM2MDc5MDAxNDddLCBbNTEuMjQwNDIyODg0MDAyMzMsIDMuMzkzMzI1MDY5OTE5MjI4Nl0sIFs1MS4yNDAzODk1MjQwMjc3MDUsIDMuMzkzMzExNzQyNjkzMTg2M10sIFs1MS4yNDAzMzk1Njc4ODQ4LCAzLjM5MzI3MDAwMDgxNTM5MTVdLCBbNTEuMjQwMjg2MTc1MTYxNiwgMy4zOTMyMjAwNDQ2NzI0ODk2XSwgWzUxLjI0MDIyOTUxMzQ5NjE2LCAzLjM5MzE4MDA2Mjk5NDM2MV0sIFs1MS4yNDAxNjEyMDA5ODUzMSwgMy4zOTMxNjY3MzU3NjgzMThdLCBbNTEuMjQwMDk3ODMzNzk3MzM2LCAzLjM5MzE3MDAwNDcxMDU1NV0sIFs1MS4yNDAwMzExOTc2NjcxMiwgMy4zOTMxODE3MzkzNzQ5OTU3XSwgWzUxLjIzOTk2OTUwNjg1OTc5LCAzLjM5MzE5NTA2NjYwMTAzODRdLCBbNTEuMjM5OTE3ODc0MzM2MjUsIDMuMzkzMjA2NzE3NDQ2NDQ2NF0sIFs1MS4yMzk4NzExODcxMzU1OCwgMy4zOTMxOTgzMzU1NDMyNzQ0XSwgWzUxLjIzOTgyMjkwNzM3MzMxLCAzLjM5MzE3NTAzMzg1MjQ1NzZdLCBbNTEuMjM5Nzg5NTQ3Mzk4NjgsIDMuMzkzMTUzNDA4NTQyMjc1XSwgWzUxLjIzOTc0NjIxMjk1OTMsIDMuMzkzMTQxNjczODc3ODM1M10sIFs1MS4yMzk2OTEyMjc2NzQ0OSwgMy4zOTMxMzAwMjMwMzI0MjczXSwgWzUxLjIzOTYzNjE1ODU3MDY0LCAzLjM5MzEwMzM2ODU4MDM0MV0sIFs1MS4yMzk1ODQ1MjYwNDcxMSwgMy4zOTMwNzAwMDg2MDU3MTg2XSwgWzUxLjIzOTU0Mjg2Nzk4ODM1LCAzLjM5MzAyODM1MDU0Njk1NTZdLCBbNTEuMjM5NTE3ODg5OTE2OSwgMy4zOTI5ODY2OTI0ODgxOTM1XSwgWzUxLjIzOTU1Mjg0MjQ1MzEyLCAzLjM5MzAwNTA0ODg1NjEzOV0sIFs1MS4yMzk1NTI4NDI0NTMxMiwgMy4zOTMwMDUwNDg4NTYxMzldLCBbNTEuMjM5NTkxMjMxNTY5NjU1LCAzLjM5MzAyMDA1MjQ2MjgxNjddLCBbNTEuMjM5NjE5NTYyNDAyMzcsIDMuMzkzMDU2NjgxMzc5Njc1NF0sIFs1MS4yMzk2NDk1Njk2MTU3MiwgMy4zOTMwOTUwNzA0OTYyMDFdLCBbNTEuMjM5Njg0NTIyMTUxOTUsIDMuMzkzMTUwMDU1NzgxMDA3M10sIFs1MS4yMzk2OTc4NDkzNzc5OSwgMy4zOTMyMTUwMTU1MzA1ODYyXSwgWzUxLjIzOTcwNDU1NDkwMDUzLCAzLjM5MzI4NjY4MDgwMjcwM10sIFs1MS4yMzk3MjQ1MDM4MzAwOCwgMy4zOTMzMzMzNjgwMDMzNjg0XSwgWzUxLjIzOTcxMTE3NjYwNDAzLCAzLjM5MzM1MTcyNDM3MTMxNF0sIFs1MS4yMzk3MDEyMDIxMzkyNiwgMy4zOTMzNDUwMTg4NDg3NzddLCBbNTEuMjM5Njg2MTk4NTMyNTgsIDMuMzkzMzIwMDQwNzc3MzI1Nl0sIFs1MS4yMzk2Nzc5MDA0NDg0NCwgMy4zOTMyNzUwMjk5NTcyOTQ1XSwgWzUxLjIzOTY2Nzg0MjE2NDY0LCAzLjM5MzI0MDA3NzQyMTA2ODddLCBbNTEuMjM5NjQyODY0MDkzMTg0LCAzLjM5MzIyNjY2NjM3NTk5NDJdLCBbNTEuMjM5NjI0NTA3NzI1MjQsIDMuMzkzMjQ2Njk5MTI0NTc0N10sIFs1MS4yMzk2MDk1MDQxMTg1NiwgMy4zOTMyOTUwNjI3MDU4NzRdLCBbNTEuMjM5NTk3ODUzMjczMTYsIDMuMzkzMzUwMDQ3OTkwNjhdLCBbNTEuMjM5NTc0NTUxNTgyMzM2LCAzLjM5MzQwODM4NjAzNjc1MzddLCBbNTEuMjM5NTQ3ODk3MTMwMjQ0LCAzLjM5MzQ2ODQwMDQ2MzQ2Ml0sIFs1MS4yMzk1MTk1NjYyOTc1NCwgMy4zOTM1Mjg0MTQ4OTAxN10sIFs1MS4yMzk0NzYyMzE4NTgxMzQsIDMuMzkzNTYzMzY3NDI2Mzk1XSwgWzUxLjIzOTQyMjgzOTEzNDkzLCAzLjM5MzU3NjY5NDY1MjQzOF0sIFs1MS4yMzkzNTQ1MjY2MjQwOCwgMy4zOTM1OTAwMjE4Nzg0ODFdLCBbNTEuMjM5Mjg2MjE0MTEzMjM1LCAzLjM5MzYwMzM0OTEwNDUyMzddLCBbNTEuMjM5MjE3OTAxNjAyMzgsIDMuMzkzNjEzNDA3Mzg4MzI5NV0sIFs1MS4yMzkxNDc4Mjg4OTE4NywgMy4zOTM2MzAwMDM1NTY2MDldLCBbNTEuMjM5MDU2MjE0NjkwMjEsIDMuMzkzNjYzMzYzNTMxMjMyM10sIFs1MS4yMzg5Njc4Njk0MzA3OSwgMy4zOTM3MDY2OTc5NzA2MjgzXSwgWzUxLjIzODg4Mjg3NjkzMjYyLCAzLjM5Mzc1MDAzMjQxMDAyNTZdLCBbNTEuMjM4ODAyODI5NzU3MzQsIDMuMzkzNzg2NzQ1MTQ1OTE3NF0sIFs1MS4yMzg3MTk1MTM2Mzk4MSwgMy4zOTM4MTUwNzU5Nzg2MzddLCBbNTEuMjM4NjM2MTk3NTIyMjgsIDMuMzkzODM4Mzc3NjY5NDUzNl0sIFs1MS4yMzg1NTQ1NTc3ODUzOSwgMy4zOTM4NjMzNTU3NDA5MDVdLCBbNTEuMjM4NDcxMTU3ODQ4ODM1LCAzLjM5Mzg4Njc0MTI1MDc1MzRdLCBbNTEuMjM4Mzg2MTY1MzUwNjc2LCAzLjM5MzkxODM0MTAyNTcxMDVdLCBbNTEuMjM4MzAxMTcyODUyNTE2LCAzLjM5Mzk0NTA3OTI5NjgyNzNdLCBbNTEuMjM4MjE2MTgwMzU0MzYsIDMuMzkzOTYzMzUxODQ1NzQxM10sIFs1MS4yMzgxMzQ1NDA2MTc0NjYsIDMuMzkzOTk1MDM1NDM5NzI5N10sIFs1MS4yMzgwNTc4NDYyMDM0NSwgMy4zOTQwMjUwNDI2NTMwODQyXSwgWzUxLjIzNzk4NDUwNDU1MDY5NSwgMy4zOTQwNjY3MDA3MTE4NDY0XSwgWzUxLjIzNzkyMTIyMTE4MTc1LCAzLjM5NDEzMDA2Nzg5OTgyMzZdLCBbNTEuMjM3ODgxMjM5NTAzNjE1LCAzLjM5NDIzMTc0MDM4NTI5MzVdLCBbNTEuMjM3ODU0NTAxMjMyNTA1LCAzLjM5NDM1MDAwOTAzOTA0NF0sIFs1MS4yMzc4Mjk1MjMxNjEwNTQsIDMuMzk0NDY2Njg1MTMxMTkyXSwgWzUxLjIzNzgyNjE3MDM5OTc4NSwgMy4zOTQ1ODUwMzc2MDM5NzQzXSwgWzUxLjIzNzgyNDQ5NDAxOTE1LCAzLjM5NDY5NTAwODE3MzU4NV0sIFs1MS4yMzc4MDYyMjE0NzAyNCwgMy4zOTQ3OTY2ODA2NTkwNTU3XSwgWzUxLjIzNzc4MjgzNTk2MDM5LCAzLjM5NDg4ODM3ODY3OTc1MjNdLCBbNTEuMjM3ODU0NTAxMjMyNTA1LCAzLjM5NDI5MTY3MDk5Mjk3MDVdLCBbNTEuMjM3ODU5NTMwMzc0NDE1LCAzLjM5NDQxNjcyODk4ODI5XSwgWzUxLjIzNzg2NjIzNTg5Njk0NSwgMy4zOTQ1MzUwODE0NjEwNzJdLCBbNTEuMjM3ODkxMjEzOTY4MzksIDMuMzk0NjgwMDA0NTY2OTA4XSwgWzUxLjIzNzg5NDU2NjcyOTY2LCAzLjM5NDc5MDA1ODk1NTU0OTddLCBbNTEuMjM3ODY5NTA0ODM5MTgsIDMuMzk0ODgxNjczMTU3MjE1XV0sCiAgICAgICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAib3JhbmdlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiBmYWxzZSwKICAiZmlsbENvbG9yIjogIm9yYW5nZSIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAibm9DbGlwIjogZmFsc2UsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInNtb290aEZhY3RvciI6IDEuMCwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfYjQ5ZTUzMzNjODM1NGJjYmE4OTdmYmJkNTdiOGVmYWEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICAgICAgCgogICAgICAgICAgICAgICAgbWFwX2I0OWU1MzMzYzgzNTRiY2JhODk3ZmJiZDU3YjhlZmFhLmZpdEJvdW5kcygKICAgICAgICAgICAgICAgICAgICBbWzUxLjIzNjY3MjkwNDM0MjQxLCAzLjM5MjY1NTAyMDU3OTY5NTddLCBbNTEuMjQwNTk5NDkwNzAyMTUsIDMuMzk1MjczMzU5NDkyNDIwN11dLAogICAgICAgICAgICAgICAgICAgIHt9CiAgICAgICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgCjwvc2NyaXB0Pg==" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
#@title Show the alignment between the 2 laps as a HTML table
from ipywidgets import *
timetable_df=timetable_df.dropna().iloc[:-1]
display(HTML(timetable_df.to_html()))
```


    HTML(value='<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></…

