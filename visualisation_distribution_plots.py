"""Auxiliary classes for visualization of distributions.
    - Histograms (e.g. of bankruptcy events by size)
    - Ensembles of distributions (e.g., of firm sizes at the end of each replication across the ensemble)
   These classes should primarily be used from and by classes defined in visualisation.py"""

from scipy.stats import expon
import numpy as np
import matplotlib.pyplot as plt
import pdb

"""Class for plots of ensembles of distributions as CDF (cumulative distribution function) or cCDF (complementary 
    cumulative distribution function) with mean, median, and quantiles"""
class CDFDistribution():
    def __init__(self, samples_x):
        """Constructor.
            Arguments:
                samples_x: list of list or ndarray of int or float - list of samples to be visualized.
            Returns:
                Class instance"""
        self.samples_x = []
        self.samples_y = []
        for x in samples_x:
            if len(x) > 0:
                x = np.sort(np.asarray(x, dtype=np.float64))
                y = (np.arange(len(x), dtype=np.float64)+1) / len(x)
                self.samples_x.append(x)
                self.samples_y.append(y)
        self.series_y = None
        self.median_x = None
        self.mean_x = None
        self.quantile_series_x = None
        self.quantile_series_y_lower = None
        self.quantile_series_y_upper = None
        
    def make_figure(self, upper_quantile=.25, lower_quantile=.75):
        #pdb.set_trace()
        """Method to do the necessary computations to create the CDF plot (incl. mean, median, quantiles.
           This method populates the variables that are plotted.
            Arguments:
                upper_quantile: float \in [0,1] - upper quantile threshold
                lower_quantile: float \in [0,1] - lower quantile threshold
            Returns None."""

        """Obtain ordered set of all y values"""
        self.series_y = np.unique(np.sort(np.hstack(self.samples_y)))

        """Obtain x coordinates corresponding to the full ordered set of all y values (self.series_y) for each series"""
        set_of_series_x = []
        for i in range(len(self.samples_x)):
            x = [self.samples_x[i][np.argmax(self.samples_y[i]>=y)] if self.samples_y[i][0]<=y else 0 for y in self.series_y]
            set_of_series_x.append(x)
            
        """Join x coordinates to matrix of size m x n (n: number of series, m: length of ordered set of y values (self.series_y))"""
        series_matrix_x = np.vstack(set_of_series_x)

        """Compute x quantiles, median, mean across all series"""
        quantile_lower_x = np.quantile(series_matrix_x,.25, axis=0)
        quantile_upper_x = np.quantile(series_matrix_x,.75, axis=0)
        self.median_x = np.quantile(series_matrix_x,.50, axis=0)
        self.mean_x = series_matrix_x.mean(axis=0)

        """Obtain x coordinates for quantile plots. This is the ordered set of all x coordinates in lower and upper quantile series."""
        self.quantile_series_x = np.unique(np.sort(np.hstack([quantile_lower_x, quantile_upper_x])))
        
        """Obtain y coordinates for quantile plots. This is one y value for each x coordinate."""
        #self.quantile_series_y_lower = [self.series_y[np.argmax(quantile_lower_x>=x)] if quantile_lower_x[0]<=x else 0 for x in self.quantile_series_x]
        self.quantile_series_y_lower = np.asarray([self.series_y[np.argmax(quantile_lower_x>=x)] if np.sum(np.argmax(quantile_lower_x>=x))>0 else np.max(self.series_y) for x in self.quantile_series_x])
        self.quantile_series_y_upper = np.asarray([self.series_y[np.argmax(quantile_upper_x>=x)] if quantile_upper_x[0]<=x else 0 for x in self.quantile_series_x])
        
        """The first value of lower must be zero"""
        self.quantile_series_y_lower[0] = 0.0
    
    def reverse_CDF(self):
        """Method to reverse the CDFs and obtain the complementary CDFs (survival functions) instead.
           The method overwrites the attributes used for plotting.
            Arguments: None.
            Returns: None."""
        self.series_y = 1. - self.series_y
        self.quantile_series_y_lower = 1. - self.quantile_series_y_lower
        self.quantile_series_y_upper = 1. - self.quantile_series_y_upper
        
    def plot(self, ax=None, ylabel="CDF(x)", xlabel="y", upper_quantile=.25, lower_quantile=.75, force_recomputation=False, show=False, outputname=None, color="C2", plot_cCDF=False):
        """Method to compile the plot. The plot is added to a provided matplotlib axes object or a new one is created.
            Arguments: 
                ax: matplitlib axes             - the system of coordinates into which to plot
                ylabel: str                     - y axis label
                xlabel: str                     - x axis label
                upper_quantile: float \in [0,1] - upper quantile threshold
                lower_quantile: float \in [0,1] - lower quantile threshold
                force_recomputation: bool       - force re-computation of plots
                show: bool                      - show plot
                outputname: str                 - output file name without ending
                color: str or other admissible matplotlib color label - color to use for the plot
                plot_cCDF: bool                 - plot survival function (cCDF) instead of CDF
            Returns: None."""
        
        """If data set is empty, return without plotting"""
        if self.samples_x == []:
            return
            
        """Create figure if none was provided"""
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        """Compute plots if not already done or if recomputation was requested"""
        if (self.series_y is None) or force_recomputation:
            self.make_figure(upper_quantile, lower_quantile)
        
        """Switch to cCDF if requested"""
        if plot_cCDF:
            self.reverse_CDF()
        
        """Plot"""
        ax.fill_between(self.quantile_series_x, self.quantile_series_y_lower, self.quantile_series_y_upper, facecolor=color, alpha=0.25)
        ax.plot(self.median_x, self.series_y, color=color)
        ax.plot(self.mean_x, self.series_y, dashes=[3, 3], color=color)
        
        """Set plot attributes"""
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        
        """Save if filename provided"""
        if outputname is not None:
            plt.savefig(outputname + ".pdf")
            plt.savefig(outputname + ".png", density=300)
        
        """Show if requested"""
        if show:
            plt.show()


"""Class for histogram plots."""
class Histogram():
    def __init__(self, sample_x):
        self.sample_x = sample_x
    
    def plot(self, ax=None, ylabel="PDF(x)", xlabel="x", num_bins=50, show=False, outputname=None, color="C2", logscale=False, xlims=None):
        """Method to compile the plot. The plot is added to a provided matplotlib axes object or a new one is created.
            Arguments: 
                ax: matplitlib axes             - the system of coordinates into which to plot
                ylabel: str                     - y axis label
                xlabel: str                     - x axis label
                num_bins: int                   - number of bins
                show: bool                      - show plot
                outputname: str                 - output file name without ending
                color: str or other admissible matplotlib color label - color to use for the plot
                logscale: bool                  - y axis logscale
                xlims: tuple, array of len 2, or none - x axis limits
            Returns: None."""
        
        """Create figure if none was provided"""
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        """Plot"""
        ax.hist(self.sample_x, bins=num_bins, color=color)

        """Set plot attributes"""
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        
        """Set xlim if requested"""
        if xlims is not None:
            ax.set_xlim(xlims[0], xlims[1])

        """Set yscale to log if requested"""
        if logscale:
            ax.set_yscale("log")
        
        """Save if filename provided"""
        if outputname is not None:
            plt.savefig(outputname + ".pdf")
            plt.savefig(outputname + ".png", density=300)
        
        """Show if requested"""
        if show:
            plt.show()


if __name__ == "__main__":
    """Unit test for CDF Distribution plot"""
    samples_x = []
    for i in range(20):
        x = expon.rvs(0.1, size=100)
        samples_x.append(x)
    
    C = CDFDistribution(samples_x)
    C.plot(upper_quantile=.25, lower_quantile=.75, show=True)
    C.plot(ylabel="cCDF(x)", plot_cCDF=True, show=True)

    """Unit test for Histogram plot"""
    x = np.sort(expon.rvs(0.1, size=10000))
    H = Histogram(x)
    H.plot(show=True)
