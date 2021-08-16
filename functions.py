import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from mne import Evoked, pick_channels


# Define subject list function
def get_sub_list(data_dir, allow_all=False, is_source=False):

    # Ask for subject IDs to analyze
    print('What IDs are being preprocessed?')
    print('(Enter multiple values separated by a comma; e.g., 101,102)')
    if allow_all:
        print('To process all subjects, type all')
    sub_list = input('Enter IDs: ')

    if sub_list == 'all' and allow_all:
        if is_source:
            sub_list = [x.name for x in data_dir.glob('sub-p3e2s*')]
        else:
            sub_list = [x.name for x in data_dir.glob('sub-*')]
    else:
        sub_list = sub_list.split(',')
        if is_source:
            sub_list = [f'sub-p3e2s{x}' for x in sub_list]
        else:
            sub_list = [f'sub-{x}' for x in sub_list]

    sub_list.sort()
    return sub_list


def _handle_picks(ch_names, picks):
    if picks is None:
        picks = np.arange(len(ch_names))
    else:
        if all(isinstance(pick, str) for pick in picks):
            picks = pick_channels(ch_names, picks)
        elif all(isinstance(pick, int) for pick in picks):
            pass
        else:
            ValueError('picks must be a list of strings or list of integers')
    return picks


def _find_nearest(a, a0, axis=-1, return_index=False):
    idx = np.abs(a-a0).argmin(axis=axis)
    if return_index:
        return idx
    else:
        return a.flat[idx]


def _get_tmin_tmax(times, tmin=None, tmax=None):
    tmin = times.min() if tmin is None else _find_nearest(times, tmin, axis=0)
    tmax = times.max() if tmax is None else _find_nearest(times, tmax, axis=0)
    return tmin, tmax


def _get_time_win(times, tmin, tmax, return_sample=False):
    tmin, tmax = _get_tmin_tmax(times, tmin=tmin, tmax=tmax)
    time_mask = np.logical_and(times >= tmin, times < tmax)
    if return_sample:
        smin = np.where(time_mask)[0][0]
        smax = np.where(time_mask)[0][-1]
        return time_mask, smin, smax
    else:
        return time_mask


def frac_area_latency(inst, mode='abs', frac=None, tmin=None, tmax=None):

    # Get Data vector and sample indicies for tmin and tmax
    ch_names = inst.info['ch_names']
    data = np.squeeze(inst.data) * 1e6
    times = inst.times
    speriod = 1 / inst.info['sfreq']
    time_win, smin, smax = _get_time_win(times, tmin=tmin, tmax=tmax,
                                         return_sample=True)

    # Process data based on mode
    if mode == 'pos':
        data[data < 0] = 0
    elif mode == 'neg':
        data[data > 0] = 0
    data = np.abs(data)  # Always rectify

    # Compute area between tmin and tmax (in time_win)
    area = trapezoid(data[:, time_win], dx=speriod, axis=1)
    if frac is None or frac == 1.0:
        return ch_names, area

    # Compute cumulative area by finding nearest 'cumulative' area
    frac_area = area * frac
    running_area = np.ones_like(data) * 10
    for i, sx in enumerate(np.arange(smin + 1, smax + 1)):
        a = trapezoid(data[:, smin:sx], dx=speriod, axis=1)
        running_area[:, smin+i] = a
    search_samples = np.arange(smin, smax+1)
    frac_lat_samples = _find_nearest(running_area[:, time_win],
                                     frac_area[:, None],
                                     axis=1, return_index=True)
    frac_true_samples = search_samples[frac_lat_samples]
    frac_lat_times = times[frac_true_samples]

    # Return computed values
    return ch_names, area, frac_lat_times


def peak_amp_lat(inst, mode='abs', tmin=None, tmax=None, picks=None,
                 adaptive=True, return_microvolts=True):
    """[summary]

    Parameters
    ----------
    inst : [type]
        [description]
    mode : str, optional
        [description], by default 'abs'
    tmin : [type], optional
        [description], by default None
    tmax : [type], optional
        [description], by default None
    picks : [type], optional
        [description], by default True
    adaptive : bool, optional
        [description], by default True
    return_microvolts : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """

    # Check inst input
    if isinstance(inst, Evoked):
        TypeError('inst must be of Evoked type')

    # Handle picks
    picks = _handle_picks(inst.ch_names, picks)
    data = inst.data
    if return_microvolts:
        data *= 1e6
    if mode not in ['neg', 'pos', 'abs']:
        ValueError("mode must be 'pos', 'neg', or 'abs'")

    # Extract times and handle tmin and tmax
    times = inst.times
    if tmin not in times and tmax not in times:
        ValueError('tmin and tmax must have values in inst.times')

    # Get the sampling rate in seconds
    srate = (1 / inst.info['sfreq'])

    # Initialize output dataframe
    out_df = pd.DataFrame(columns=['ch_name', 'tmin', 'tmax',
                                   'peak_amplitude', 'peak_latency'])

    # Loop through channels
    for i, pick in enumerate(picks):

        # Reset ch_tmin and ch_tmax
        ch_tmin = tmin
        ch_tmax = tmax

        # Find the peak latency and amplitude
        while True:

            # Get time window for this iteration
            time_mask = np.logical_and(times >= ch_tmin, times <= ch_tmax)
            time_window = times[time_mask]

            # Extract windowed data
            data_window = data[pick, time_mask]
            sign_window = np.sign(data_window)
            if mode == 'neg':
                data_window[data_window > 0] = 0
            elif mode == 'pos':
                data_window[data_window < 0] = 0
            data_window = np.abs(data_window)

            # Extract peak information
            peak_lat_index = np.argmax(data_window)
            peak_latency = time_window[peak_lat_index]
            peak_amplitude = data_window[peak_lat_index]
            peak_sign = sign_window[peak_lat_index]

            # If adaptive, adjust window
            if adaptive:
                if peak_latency == ch_tmin:
                    ch_tmin += -srate
                    ch_tmax += -srate
                elif peak_latency == ch_tmax:
                    ch_tmin += srate
                    ch_tmax += srate
                else:
                    break
            else:
                break

        # Add to output
        out_df.at[i, :] = [ch_names[pick], ch_tmin, ch_tmax,
                           peak_amplitude * peak_sign, peak_latency]

    # Return
    return out_df


def mean_amplitude(inst, tmin=None, tmax=None, picks=None,
                   return_microvolts=True):
    """[summary]

    Parameters
    ----------
    inst : [type]
        [description]
    tmin : [type], optional
        [description], by default None
    tmax : [type], optional
        [description], by default None
    picks : [type], optional
        [description], by default None
    return_microvolts : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    # Check inst input
    if isinstance(inst, Evoked):
        TypeError('inst must be of Evoked type')

    # Handle picks
    ch_names = inst.ch_names
    if picks is None:
        picks = np.arange(len(ch_names))
    else:
        if all(isinstance(pick, str) for pick in picks):
            picks = pick_channels(ch_names, picks)
        elif all(isinstance(pick, int) for pick in picks):
            pass
        else:
            ValueError('picks must be a list of strings or list of integers')

    # Extract times and handle tmin and tmax
    if tmin not in inst.times and tmax not in inst.times:
        ValueError('tmin and tmax must have values in inst.times')
    time_mask = np.logical_and(inst.times >= tmin, inst.times <= tmax)

    # Initialize output dataframe
    out_df = pd.DataFrame(columns=['ch_name', 'tmin', 'tmax',
                                   'mean_amplitude'])

    # Loop through channels
    for i, pick in enumerate(picks):

        # Get mean amplitude
        mean_amp = inst.data[pick, time_mask].mean(axis=-1)
        if return_microvolts:
            mean_amp *= 1e6

        # Add to output
        out_df.at[i, :] = [ch_names[pick], tmin, tmax, mean_amp]

    # Return
    return out_df
