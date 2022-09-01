import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cooltools import numutils
import cv2

def get_compartment_bins(eigenvector):
    """
    Get indexes of compartments A/B and NaN (zero) bins from eigenvector.
    Parameters:
    eigenvector -- A list with eigenvector values for specific chromosome.
    Output:
    Three lists with indexes of compartment A/B and zeros bins.
    """

    compartment_A = [ind for (ind, eig) in zip(np.arange(len(eigenvector)), eigenvector) if eig > 0]
    compartment_B = [ind for (ind, eig) in zip(np.arange(len(eigenvector)), eigenvector) if eig < 0]
    zero_bins = [ind for (ind, eig) in zip(np.arange(len(eigenvector)), eigenvector) if eig == 0]
    return(compartment_A, compartment_B, zero_bins)

def calculate_intervals_from_range(list_range):
    """
    Merge a list of continuous indexes into list of intervals.
    Parameters:
    list_range -- A list with compartment indexes to be merged into intervals.
    Output:
    A list of lists with intervals.
    """

    list_range = list(list_range)
    intervals = []
    for idx, item in enumerate(list_range):
        if not idx or item-1 != intervals[-1][-1]:
            intervals.append([item])
        else:
            intervals[-1].append(item)
    return(intervals)

def get_compartment_intervals(compartment_A, compartment_B, zero_bins):
    """
    Apply intervals merging to compartment A/B and zeros bins.
    Parameters:
    compartment_A -- indexes of eigenvector corresponding to compartment A bins.
    compartment_B -- indexes of eigenvector corresponding to compartment B bins.
    zero_bins -- indexes of eigenvector corresponding to zero bins.
    Output:
    Intervals of compartment A/B and zeros bins.
    """

    intervals_A = calculate_intervals_from_range(compartment_A)
    intervals_B = calculate_intervals_from_range(compartment_B)
    intervals_zero = calculate_intervals_from_range(zero_bins)
    return(intervals_A, intervals_B, intervals_zero)

def get_area_from_matrix(matrix, intervals_list_1, intervals_list_2):
    """
    Extract an area from Hi-C map that lies at the intersection of two intervals.
    Parameters:
    matrix -- Full Hi-C map of a chromosome.
    intervals_list_1 -- First interval for extraction.
    intervals_list_2 -- Second interval for extraction.
    Output:
    Extracted Hi-C map area.
    """

    return(matrix[np.ix_(intervals_list_1, intervals_list_2)])

def area_dimensions_are_large_enough(img, min_dimension):
    """
    Find whether extracted area is large enough for the analysis.
    Parameters:
    img -- Hi-C map area extracted.
    min_dimensions -- Minimum size for area dimensions.
    Output:
    Whether an area is large enough.
    """

    return(img.shape[1] >= min_dimension and img.shape[0] >= min_dimension)

def area_has_enough_data(img, max_zeros_fraction):
    """
    Find whether extracted area has enough data for the analysis.
    Parameters:
    img -- Hi-C map area extracted.
    max_zeros_fraction -- Maximum fraction of zeros in area.
    Output:
    Whether an area has enough data.
    """

    return(len(np.where(img.ravel() == 0)[0]) < max_zeros_fraction * len(img.ravel()))

def area_is_close_enough(intervals_1, intervals_2, matrix_size, cutoff):
    """
    Find whether extracted area has enough data for the analysis.
    Parameters:
    intervals_1 -- First interval used for extraction.
    intervals_2 -- Second interval used for extraction.
    matrix_size -- Size of full Hi-C map of chromosome.
    cutoff -- Maximum distance between intervals as chromosome size fraction.
    Output:
    Whether an area is close enough to the diagonal.
    """

    return(np.mean(intervals_2) < np.mean(intervals_1) + cutoff * matrix_size)

def resize_area(img, bin_size):
    """
    Rescale Hi-C map area to a square with defined edge size.
    Parameters:
    img -- Hi-C map area extracted.
    bin_size -- Rescaling sequare size in bins.
    Output:
    Rescaled Hi-C map area.
    """

    img_resized = cv2.resize(img * 255 / max(img.ravel()), (bin_size, bin_size))
    img_resized = img_resized / 255 * max(img.ravel())
    return(img_resized)

def get_area_type(interval_1, interval_2, intervals_A, intervals_B):
    """
    Find what type of area was extracted from the Hi-C map.
    Parameters:
    interval_1 -- First interval used for extraction.
    interval_2 -- Second interval used for extraction.
    intervals_A -- List of compartment A intervals.
    intervals_B -- List of compartment B intervals.
    Output:
    Area type.
    """

    if (interval_1 in intervals_A and interval_2 in intervals_B) or\
       (interval_1 in intervals_B and interval_2 in intervals_A):
        return('AB')
    elif (interval_1 in intervals_A and interval_2 in intervals_A):
        return('A')
    elif (interval_1 in intervals_B and interval_2 in intervals_B):
        return('B')
    
def plot_pentads(average_compartment, title, vmax=2.0, vmin=0.5, cmap='coolwarm', out_pref=None):
    subplot_titles = ['Short-range A', 'Short-range B',
                      'Long-range A', 'Long-range B',
                      'Between A and B']
    subplot_indexes = [4, 8, 6, 2, 5]

    fig = plt.figure(figsize = (10, 10))
    plt.suptitle(title, x = 0.5125, y = 0.98, fontsize = 22)

    for subtitle, index in zip(subplot_titles, subplot_indexes):
        plt.subplot(3, 3, index)
        plt.title(subtitle, fontsize = 15)
        plt.imshow(average_compartment[subtitle], cmap = cmap, norm = LogNorm(vmax = vmax, vmin = vmin))
        plt.xticks([], [])
        plt.yticks([], [])

    cbar_ax = fig.add_axes([0.95, 0.25, 0.02, 0.5])
    cbar = plt.colorbar(cax = cbar_ax)

    if out_pref is not None:
        plt.savefig(out_pref + '.png', bbox_inches = 'tight')
    plt.show()
    
def get_comp_quantiles(AVG, q1=0.1, q2=0.9):
    """get minimum and maximum values for vizualization"""
    q1s, q3s = [], []
    for _, ac in AVG.items():
        #average compartment statistics
        for k in ac.keys():
            m = ac[k].ravel()
            q1s.append(np.quantile(m, q1))
            q3s.append(np.quantile(m, q2))
    return np.min(q1s), np.max(q3s)

def plot_pentads_group(AVG, chrom, vmax=2.0, vmin=0.5, out_pref=None):
    from matplotlib.colors import LogNorm
    import matplotlib as mpl
    #plotting
    subplot_titles = ['Short-range A', 'Short-range B',
                      'Long-range A', 'Long-range B',
                      'Between A and B']
    subplot_indexes = [3, 7, 5, 1, 4]

    fig = plt.figure(figsize=(14, 14), constrained_layout=True)
    fig.suptitle(chrom, fontsize=20)

    subfigs = fig.subfigures(2, 2, wspace=0.1, hspace=0.1)
    for (name, average_compartment), subfig in zip(AVG.items(), subfigs.flat):
        title = name #subfigure title
        subfig.suptitle(title, fontsize=16)
        axs = subfig.subplots(3, 3).flatten()
        for subtitle, index in zip(subplot_titles, subplot_indexes):
            axs[index].set_title(subtitle, fontsize = 13)
            axs[index].imshow(average_compartment[subtitle], 
                      cmap = 'coolwarm', 
                      norm = LogNorm(vmax = vmax, vmin = vmin))
            axs[index].set_xticks([])
            axs[index].set_yticks([])
        for index in range(9): #erase 
            if index not in subplot_indexes:
                axs[index].set_visible(False)

    norm = LogNorm(vmax = vmax, vmin = vmin)
    cbar_ax = fig.add_axes([1.03, 0.25, 0.02, 0.5])
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=mpl.cm.coolwarm, norm=norm,)

    if out_pref is not None:
        plt.savefig(out_pref + '.pdf', format='pdf', bbox_inches = 'tight')
    fig.show()