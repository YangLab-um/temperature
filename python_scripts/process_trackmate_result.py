import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks, savgol_filter

min_ratio = 0.35
max_ratio = 1.55
smoothing_window = 9
smoothing_order = 2
min_prominence = 0.05
fret_channel = 'CH1'

position_list = [i for i in range(28)] # + [i + 26 for i in range(13)]
date = "08-23-24"
data_location = rf"Z:\Users\Franco\Experiments\{date}\Tracking_Result" # rf"Z:\Users\Franco\Experiments\{date}\Tracking_Result"
# save_location = rf"Y:\users\franco_tavella\temperature\raw_data\{date}"
save_location = rf"E:\Project 6 - Temperature\Experiments\data_analysis\{date}"

def detrend(x: np.array, y: np.array) -> np.array:
    not_nan_ind = ~np.isnan(y)
    m, b, _, _, _ = linregress(x[not_nan_ind], y[not_nan_ind])
    detrend_y = y - (m*x + b)
    final_y = detrend_y + np.nanmean(y)
    return final_y

for pos in position_list:
    print(f"Processing position {pos}")
    # Read in the data
    spots = pd.read_csv(f"{data_location}/Pos{pos}_spots.csv", encoding='cp1252', skiprows=range(1, 4))
    tracks = pd.read_csv(f"{data_location}/Pos{pos}_tracks.csv", encoding='cp1252', skiprows=range(1, 4))
    bit_to_ratio = lambda x: x * (max_ratio - min_ratio) / 65535 + min_ratio
    spots[f'MEAN_INTENSITY_{fret_channel}'] = spots[f'MEAN_INTENSITY_{fret_channel}'].apply(bit_to_ratio)
    # Storage for the data
    processed_spots = pd.DataFrame(columns=['TRACK_ID', 'TIME', 'RATIO'])
    peaks_and_troughs = pd.DataFrame(columns=['TRACK_ID', 'TIME', 'RATIO', 'TYPE', 'CYCLE'])
    # Iterate over each track, detrend the data, and find peaks and troughs
    all_ids = spots['TRACK_ID'].unique()
    for id in all_ids:
        track = spots[spots['TRACK_ID'] == id]
        # Order by time, reset index
        track = track.sort_values(by='POSITION_T').reset_index(drop=True)
        x = track['POSITION_T']
        y = track[f'MEAN_INTENSITY_{fret_channel}']
        y_detrended = detrend(x, y)
        y_smooth = savgol_filter(y_detrended, smoothing_window, smoothing_order)
        x_peaks, peak_properties = find_peaks(y_smooth, prominence=min_prominence)
        x_troughs, trough_properties = find_peaks(-y_smooth, prominence=min_prominence)
        # Save detrended data
        track_df = pd.DataFrame({'TRACK_ID': [id]*len(x), 'TIME': x, 'RATIO': y_detrended,
                                 'POSITION_X': track['POSITION_X'], 'RADIUS': track['RADIUS']})
        processed_spots = pd.concat([processed_spots, track_df], ignore_index=True)
        # Save peaks and troughs
        peaks_df = pd.DataFrame({'TRACK_ID': [id]*len(x_peaks), 'TIME': x[x_peaks],
                                 'RATIO': y_detrended[x_peaks], 'TYPE': ['PEAK']*len(x_peaks),
                                 'CYCLE': np.arange(1, len(x_peaks)+1)})
        troughs_df = pd.DataFrame({'TRACK_ID': [id]*len(x_troughs), 'TIME': x[x_troughs],
                                   'RATIO': y_detrended[x_troughs], 'TYPE': ['TROUGH']*len(x_troughs),
                                   'CYCLE': np.arange(1, len(x_troughs)+1)})
        peaks_and_troughs = pd.concat([peaks_and_troughs, peaks_df], ignore_index=True)
        peaks_and_troughs = pd.concat([peaks_and_troughs, troughs_df], ignore_index=True)
    # Save the data
    processed_spots.to_csv(f"{save_location}/Pos{pos}_processed_spots.csv", index=False)
    peaks_and_troughs.to_csv(f"{save_location}/Pos{pos}_peaks_and_troughs.csv", index=False)