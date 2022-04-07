from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from scipy import ndimage
import matplotlib.cm as cm
import numpy as np

def plot_hic(matrix, start, fin, profile=None, profnames=None, savepath=None, chrom_start=0):
    vmin, vmax = matrix.min(), matrix.max()
    n = len(profile) if profile is not None else 1
    height_ratios = [4] + [1]*n
    
    plt.figure(figsize=(18, 9 + 3*n))
    gs = GridSpec(1+n, 1, height_ratios=height_ratios, hspace=0.03)
    
    axh = plt.subplot(gs[0,0])
    rotated_img = ndimage.rotate(matrix[start-chrom_start:fin-chrom_start,
                                        start-chrom_start:fin-chrom_start], 45, cval=np.nan)
    hlimits = np.linspace(start, fin, rotated_img.shape[0])
    axh.imshow(rotated_img[:rotated_img.shape[0]//2], 
               interpolation='None', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axh.axis('off')
    
    colors=cm.rainbow(np.linspace(0,1,n))
    for i, prof, name in zip(range(n), profile, profnames):
        axp = plt.subplot(gs[i+1,0])
        prof = np.array(prof)[start:fin]
        limits = np.linspace(start, fin, prof.shape[0])
        axp.plot(limits, prof, label=name, c=colors[i])
        if i == 0:
            axp.xaxis.tick_top()
        else:
            axp.set_xticks([])
        axp.tick_params(labelsize=12)
        axp.set_xlim((limits[0], limits[-1]))
        axp.grid(alpha=0.5)
        axp.legend(fontsize=12)
    
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.show()