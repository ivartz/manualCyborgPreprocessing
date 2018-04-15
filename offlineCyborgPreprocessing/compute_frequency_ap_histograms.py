from __future__ import division, print_function
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import stft, get_window
import glob
import os

# The preprocessed output from this script should be comparable
# to the preprocessed output from Example 2 in fs2CyborgPreprocessing 
# README.md https://github.com/ivartz/fs2CyborgPreprocessing/blob/master/README.md .
#
# The main difference:
# offline preprocessing (this script): Exact peaks are counted in frequency bins.
# real-time preprocessing (fs2CyborgPreprocessing): Number of exceeds above
# the pre-computed tresholds are counted in frequency bins.
#
# Although this main difference, the relative differences in amplitudes
# in the preprocessed data might lead to similar PCA 
# models on preprocessed data from compute_frequency_ap_histograms.py
# and fs2CyborgPreprocessing Example 2 . This can help in variable selection
# described below in goal 2.
#
# In addition, this script counts in 10 Hz frequency bins in the range 300 - 3000 Hz ,
# while Example 2 in fs2CyborgPreprocessing counts in 8 Hz frequency bins in
# the range 296 - 3000 Hz .
#
# --> Main goals of this preprocessing:
# 1. Detect clusters in score plots by looking at them, 
# with algorithms such as k-means clustering. Additionally
# use MVA results from multiple MEAs in new models such 
# as hierarchical models.
#
# 2. Preprocess for multivariate analysis
# for variable  selection (frequency bin in any electrode)
# for chosing bins to listen to in real-time analysis in SHODAN.
#
# Define directory with noise reduced wav files (noise reduced with Audacity).
raw_filenames = sorted(glob.glob("/media/loek/HD/Cyborg/Master thesis/Sound/2017-04-24T10-45-43 Noise Reduced/*.wav"))
#
# Define output directory to store individual and combined aggregated peak count histograms.
output_directory = "/media/loek/HD/Cyborg/Master thesis/data/Preprocessed/offline/2017-04-24T10-45-43"
# First part is a custom preak detection algorithm written by Marcos Duarte:
# ---- start peak detection algorithm
# script motivated by matlab's https://se.mathworks.com/help/signal/ref/findpeaks.html
# http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

"""Detect peaks in data based on their amplitude and other features."""

__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.4"
__license__ = "MIT"

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()
# ---- end peak detection algorithm
# The rest of the script constructs aggregated counts (histograms)
# of peaks in action potentials in 6 second sliding windows,
# overlapping 80 % .

# Read first file in order to get total length.
rate, raw_data = read(raw_filenames[0])
total_duration_recording_seconds = len(raw_data)/rate

# Define sliding window.
window_start_seconds = 0
window_duration_seconds = 6

window_overlap = 0.8

window_offsets = np.arange(0, total_duration_recording_seconds - window_duration_seconds, window_duration_seconds*(1-window_overlap))

# Declare the matrix to hold the histograms.
MEA_histograms = np.zeros((window_offsets.size, 0), dtype="int32")

def extract_electrode_name_from_path(path):
    # Extract the electrode name from the file paths
    endIndex = len(path) - path.find(")") - 6
    if "Ref" in path:
        startIndex = len(path) - path.find("(") + 4
    else:
        startIndex = len(path) - path.find("(") + 3
    return path[-startIndex:-endIndex]

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over electrodes. Electrode numbering as in
# Figure 4.1 a) in project report:
# http://folk.ntnu.no/ivarthov/PCA%20on%20a%20time%20series%20of%20MEA%20recordings%20of%20Dopaminergic%20neurons%20-%20ivarthoklehovden%20-%2020171217.pdf
# The correct electrode numbering is preserved based in file names of the 
# noise reduced wav files.

for path in raw_filenames:
    
    name = extract_electrode_name_from_path(path)

    print("electrode " + name)

    rate, raw_data = read(path)

    number_of_bins = 270 # = x.size

    final_electrode_firing_rate_histogram = np.zeros((window_offsets.size, number_of_bins), dtype="int32")

    for index, current_window_offset in enumerate(window_offsets):

        print("electrode " + name + " window offset [s] " + str(current_window_offset))

        current_window_start_seconds = window_start_seconds + current_window_offset
        current_window_end_seconds = window_duration_seconds + current_window_offset

        print("electrode " + name + " window offset [s] " + str(current_window_offset) + " compute STFT")

        f, t, Zxx = stft(raw_data[int(np.floor(current_window_start_seconds*rate)):int(np.floor(current_window_end_seconds*rate))], fs=rate, window="hamming", nperseg=1000, noverlap = 1000*0.8)

        f_step = int(f[1]-f[0])
        start_f_hist = 300
        stop_f_hist = 3000

        data = np.abs(Zxx)
        firing_histogram = np.zeros(len(f), dtype={"names":["bin", "count"], "formats":["U10","i4"]})

        amp_treshold = 0.002

        print("electrode " + name + " window offset [s] " + str(current_window_offset) + " count peaks")

        #f_detected = np.zeros(len(data[:,0]))
        for f_index, row in enumerate(data):
            firing_histogram[f_index]["bin"] = str(int(f[f_index])) + "-" + str(int(f[f_index]) + f_step)

            possible_peak_indices = detect_peaks(row)

            for peak_index in possible_peak_indices:
                if row[peak_index] > amp_treshold:
                    # we are certain that it is a peak
                    firing_histogram[f_index]["count"] += 1

                    #if row[peak_index] > f_detected[index]:
                    #    f_detected[peak_index] = row[peak_index]

        start = int(np.argwhere(firing_histogram["bin"] == "300-310"))
        stop = int(np.argwhere(firing_histogram["bin"] == "3000-3010"))

        x = firing_histogram[start:stop]["bin"]
        y = firing_histogram[start:stop]["count"]

        final_electrode_firing_rate_histogram[index] = y

    print("electrode " + name + " stack histograms to MEA histograms")

    MEA_histograms = np.hstack((MEA_histograms, final_electrode_firing_rate_histogram))

    print("electrode " + name + " save histograms to disk")

    np.savetxt(output_directory + "/" + name \
               + "_AP_count_histograms" + ".csv", final_electrode_firing_rate_histogram, fmt="%i", delimiter=";", header=";".join(x))

print("MEA save stacked histograms to disk")

np.savetxt(output_directory + "/" + "Combined_AP_count_histograms.csv", \
           MEA_histograms, fmt="%i", delimiter=";", header=";".join(b + " " + e for e in [ extract_electrode_name_from_path(path) for path in raw_filenames ] for b in x))

