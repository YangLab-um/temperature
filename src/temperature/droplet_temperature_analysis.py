import numpy as np

def calculate_FRET_ratio_spatial_average(CFP_image, FRET_image, window_size):
    """
    Vertical average of the FRET ratio for several horizontal windows
    """
    # Calculate the FRET ratio
    FRET_ratio = FRET_image / CFP_image
    # Calculate the spatial average of the FRET ratio for each horizontal window
    window_starting_positions = np.arange(0, FRET_ratio.shape[1] - window_size, window_size)
    FRET_ratio_spatial_average = []
    for i in window_starting_positions:
        mean_value = np.nanmean(FRET_ratio[:, i:i + window_size])
        FRET_ratio_spatial_average.append(mean_value)
    return FRET_ratio_spatial_average
