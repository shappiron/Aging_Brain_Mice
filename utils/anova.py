import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_selection import f_classif
from statsmodels.stats.multitest import multipletests as fdr_correction


#Thic block is for ANOVA + shuffle analysis to define significant features in data. 
def pvals(X, y, thr=0.05, method='fdr_bh'):
    """This function do ANOVA and FDR correction for given X,y"""
    F_scores, P_vals = f_classif(X, y)
    P_vals_corrected = fdr_correction(P_vals, alpha=thr, method=method)
    return P_vals, P_vals_corrected

def myshuffle(y):
    """This function do a random permutation of y showing that chosen features are not random"""
    y = np.array(y)
    p = np.random.permutation(y)
    while not any(p != y):
        p = np.random.permutation(y)
    return p

def bootstrap(X, y, n=1000, thr=0.05, method='fdr_bh'):
    """This function do a random permutation test n times building 
       a statistics of random and original analysis"""
    result = []
    for _ in range(n):
        p = myshuffle(y)
        _, P_vals_corrected = pvals(X, p, thr=thr, method=method)
        result.append(sum(P_vals_corrected[0]))
    return np.mean(result), np.std(result)

#CLASS for ANOVA
class ANOVA:
    def __init__(self, X, y, 
                 p_value_threshold=0.05, method='fdr_bh'):
        assert isinstance(X, pd.core.frame.DataFrame), "X must be pd.core.frame.DataFrame type"
        #assert isinstance(y, np.ndarray), "y must be np.array type"     
        self.thr = p_value_threshold
        self.method = method
        self.X = X.copy()
        self.y = y.copy()
        self.pvals = None
        self.pvals_corrected = None
        
    def make_anova(self):
        self.pvals, self.pvals_corrected = pvals(X=self.X, y=self.y, thr=self.thr, method=self.method)
        print("Total number of features: %d" % len(self.pvals))
        print("Number of p_values <= %.3f: %d" % (self.thr, sum(self.pvals <= self.thr)))
        print("Number of p_values after FDR correction: %d" % sum(self.pvals_corrected[0]))
        return self.pvals, self.pvals_corrected
    
    def get_passed_features(self, corrected=True):
        """return the features passed p_value threshold with FDR correction (or NOT)"""
        if corrected:
            return self.X.iloc[:, self.pvals_corrected[0]]
        else:
            return self.X.iloc[:, self.pvals <= self.thr]
    
    def get_filter_distribution(self, func='FC', **kwargs):
        """ Plot a distribution as a result of applying filtration function to data
            func::function or str, defaul:`FC` means to use Fold Change between data classes"""
        if func == 'FC':
            funcname = 'Abs Fold Change'
            z = self.X.groupby(self.y).mean()
            vals = np.abs(np.array(z.iloc[0,:] - z.iloc[1,:]))
        else:
            funcname = func.__name__
            vals = np.array(X.apply(func, axis=0, **kwargs))
        #plot
        plt.figure(figsize=(9,5))
        plt.title('Distribution of IS %s for all bins' % funcname, fontsize=16)
        sns.kdeplot(vals)
        plt.grid(alpha=0.3)
        plt.show()

    def make_permutation_search(self, func, interval=(None, None), steps=10, N=20, **kwargs):
        """ Find best possible number of p-values by columns filtering
            making permutation test over y for each subset of features defined with 
            func criterion within interval of search.
            func::function or str, defaul:`FC` means to use Fold Change between data classes"""
        assert interval!=(None, None), "Please, set the interval of search."
        self.count_pcor = []
        self.count_pcor_sh_m = []
        self.count_pcor_sh_s = []
        self.S = np.linspace(interval[0], interval[1], steps)
        
        if func == 'FC':
            self.funcname = 'Abs Fold Change'
            z = self.X.groupby(self.y).mean()
            self.X_criterion = np.abs(np.array(z.iloc[0,:] - z.iloc[1,:]))
        else:
            self.funcname = func.__name__
            self.X_criterion = np.array(X.apply(func, axis=0, **kwargs))
            
        for s in tqdm(self.S):
            cols = (self.X_criterion >= s)
            X_subset = self.X.loc[:, cols]
            
            P_vals, P_vals_corrected = pvals(X_subset, self.y, thr=self.thr, method=self.method)
            P_sh_mean, P_sh_std = bootstrap(X_subset, self.y, n=N, thr=self.thr, method=self.method)

            self.count_pcor.append(sum(P_vals_corrected[0]))
            self.count_pcor_sh_m.append(P_sh_mean)
            self.count_pcor_sh_s.append(P_sh_std)
    
    def plot_permutation_results(self, savename=None):
        plt.figure(figsize=(9,6))
        plt.plot(self.S, self.count_pcor, label='True labels count')
        plt.plot(self.S, self.count_pcor_sh_m, label='Shuffled labels with mean + $\sigma$')
        plt.fill_between(self.S, self.count_pcor_sh_s, alpha=0.3, color='orange', lw=0)
        plt.title('Benjamini/Hochberg  (non-negative) corrected \n with P-value threshold = {0:.2f}'.format(self.thr) )
        plt.ylabel('Count', fontsize=12)
        plt.xlabel(self.funcname)
        plt.legend()
        if savename is not None:
            plt.savefig(savename, dpi=200, bbox_inches='tight')
        plt.show()
        
    def get_best_criterion(self):
        """ return best criterion within the interval of permutation search,
            columns criterion (according to func) must be >= best one"""
        return self.S[np.argmax(self.count_pcor)]
    
    def get_best_subset(self):
        return self.X.loc[:, (self.X_criterion >= self.get_best_criterion())]
    
    def get_passed_features_for_some_criterion(self, criterion, corrected=True, return_pvals=False):
        """return features for some threshold according to the precomputed with func"""
        X_subset = self.X.loc[:, (self.X_criterion >= criterion)]
        P_vals, P_vals_corrected = pvals(X_subset, self.y, thr=self.thr, method=self.method)
        if corrected:
            if return_pvals:
                return X_subset.iloc[:, P_vals_corrected[0]], P_vals_corrected, P_vals
            else:
                return X_subset.iloc[:, P_vals_corrected[0]]
        else:
            if return_pvals:
                return X_subset.iloc[:, P_vals <= self.thr], P_vals_corrected, P_vals
            else:
                return X_subset.iloc[:, P_vals <= self.thr]
            