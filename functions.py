import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
import scipy.io as spio

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


# Functions to read in .mat file
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


# Define EEG/ERP Functions
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


def peak_amp_lat(inst, mode='pos', tmin=None, tmax=None, picks=None,
                 return_microvolts=True, width=2, frac_peak=None):
    """Measure peak amplitude, latency, and fractional peak onset. This
    can be run on ERPs for peak latency and amplitude. Fractional peak
    onset is better conducted on difference waves. Note fractional peak
    onset is only returned if frac_peak is a float between 0 and 1.

    Uses :func:~scipy.signal.find_peaks to locate peaks.

    Parameters
    ----------
    inst : :class:~mne.Evoked object
        A single instance of an :class:~mne.Evoked object.
    mode : {‘pos’, ‘neg’, ‘abs’} (defaults 'pos')
        Controls whether positive ('pos'), negative ('neg') or absolute
        ('abs') peaks are detected. 'pos' searches for a postiive going peak
        (but the peak can take on a negative voltage). 'neg' searches for
        negative going peaks by scaling the voltages by -1 (but the peaks
        can take on a positive voltage. 'abs' finds the largest peak
        regardless of sign.
    tmin : float | None (defaults None)
        The minimum point in time to be considered for peak getting. If None
        (default), the beginning of the data is used.
    tmax : float | None (defaults None)
        The maximum point in time to be considered for peak getting. If None
        (default), the end of the data is used.
    picks : str|list|int|None (defaults None)
        Channels to include. integers and lists of integers will be interpreted
        as channel indices. str and lists of strings will be interpreted as
        channel names.
    return_microvolts : bool (defaults True)
        If True, returns the peak amplitude in μV.
    width : int|ndarray|list (defaults 2)
        Required width of peaks in samples. An integer is treated as the
        minimal required width (with no maximum). A ndarray or list of
        integers specifies the minimal and maximal widths, respectively.
    frac_peak : float [0, 1]|None (defaults None)
        If a float value, returns the latency where the voltage falls below
        frac_peak * peak_amplitude. If None, fractional peak latency is not
        returned.

    Returns
    -------
    data : instace of :class:~pandas.DataFrame
        A :class:~pandas.DataFrame with the peak amplitude, latency,
        fractional peak latency, tmin, and tmax for each channel
        specified by picks.
    """

    # Check inst input
    if isinstance(inst, Evoked):
        TypeError('inst must be of Evoked type')

    # Check frac_peak
    if frac_peak is not None:
        if frac_peak < 0 or frac_peak > 1:
            ValueError('frac_peak must be float between 0 and 1')

    # Check mode
    if mode not in ['neg', 'pos', 'abs']:
        ValueError("mode must be 'pos', 'neg', or 'abs'")

    # Handle picks
    if isinstance(picks, int) or isinstance(picks, str):
        picks = [picks]
    picks = _handle_picks(inst.ch_names, picks)

    # Extract data
    data = inst.data
    if return_microvolts:
        data *= 1e6

    # Extract times and handle tmin and tmax
    times = inst.times
    if tmin not in times and tmax not in times:
        ValueError('tmin and tmax must have values in inst.times')

    # Initialize output dataframe
    out_df = pd.DataFrame(columns=['ch_name', 'tmin', 'tmax',
                                   'peak_amplitude', 'peak_latency'
                                   'frac_peak', 'frac_peak_onset',
                                   'frac_peak_amplitude'])

    # Loop through channels
    for i, pick in enumerate(picks):

        # Get time window for this iteration
        time_mask = np.logical_and(times >= tmin, times <= tmax)
        time_window = times[time_mask]

        # Extract windowed data and manipulate as needed
        data_window = data[pick, time_mask]
        sign_window = np.sign(data_window)
        if mode == 'neg':
            data_window *= -1
        elif mode == 'abs':
            data_window = np.abs(data_window)

        # Find the peak indices and amplitudes
        peaks, _ = find_peaks(data_window)
        amplitudes = data_window[peaks]

        # Extract peak information
        peak_index = peaks[np.argmax(amplitudes)]
        peak_latency = time_window[peak_index]
        peak_amplitude = np.abs(data_window[peak_index])
        peak_amplitude *= sign_window[peak_index]

        # Search for fractional peak onset
        if frac_peak is not None:
            frac_index = peak_index
            frac_amplitude = peak_amplitude
            frac_p = np.abs(frac_amplitude / peak_amplitude)
            while frac_p > frac_peak:
                frac_index -= 1
                frac_amplitude = data[frac_index]
                frac_p = np.abs(frac_amplitude / peak_amplitude)
            frac_peak_onset = times[frac_index]
        else:
            frac_peak = 'n/a'
            frac_peak_onset = 'n/a'
            frac_peak_amplitude = 'n/a'

        # Add to output
        out_df.at[i, :] = [inst.ch_names[pick], tmin, tmax,
                           peak_amplitude, peak_latency, frac_peak,
                           frac_peak_onset, frac_peak_amplitude]

    # Return
    return out_df


def mean_amplitude(inst, tmin=None, tmax=None, picks=None,
                   return_microvolts=True):
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
