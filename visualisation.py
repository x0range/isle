# file to visualise data from a single and ensemble runs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import visualisation_distribution_plots
import argparse
import pdb
import isleconfig

class TimeSeries(object):
    def __init__(self, series_list, title="",xlabel="Time", colour='k', axlst=None, fig=None, percentiles=None, alpha=0.7):
        self.series_list = series_list
        self.size = len(series_list)
        self.xlabel = xlabel
        self.colour = colour
        self.alpha = alpha
        self.percentiles = percentiles
        self.title = title
        self.timesteps = [t for t in range(len(series_list[0][0]))] # assume all data series are the same size
        if axlst is not None and fig is not None:
            self.axlst = axlst
            self.fig = fig
        else:
            self.fig, self.axlst = plt.subplots(self.size,sharex=True)

        #self.plot() # we create the object when we want the plot so call plot() in the constructor

    def plot(self):
        for i, (series, series_label, fill_lower, fill_upper) in enumerate(self.series_list):
            self.axlst[i].plot(self.timesteps, series,color=self.colour)
            self.axlst[i].set_ylabel(series_label)
            if fill_lower is not None and fill_upper is not None:
                self.axlst[i].fill_between(self.timesteps, fill_lower, fill_upper, color=self.colour, alpha=self.alpha)
        self.axlst[self.size-1].set_xlabel(self.xlabel)
        self.fig.suptitle(self.title)
        return self.fig, self.axlst

    def save(self, filename):
        self.fig.savefig("{filename}".format(filename=filename))
        return

class InsuranceFirmAnimation(object):
    '''class takes in a run of insurance data and produces animations '''
    def __init__(self, data):
        self.data = data
        self.fig, self.ax = plt.subplots()
        self.stream = self.data_stream()
        self.ani = animation.FuncAnimation(self.fig, self.update, repeat=False, interval=100,)
                                           #init_func=self.setup_plot)

    def setup_plot(self):
        # initial drawing of the plot
        casharr,idarr = next(self.stream)
        self.pie = self.ax.pie(casharr, labels=idarr,autopct='%1.0f%%')
        return self.pie,

    def data_stream(self):
        # unpack data in a format ready for update()
        for timestep in self.data:
            casharr = []
            idarr = []
            for (cash, id, operational) in timestep:
                if operational:
                    casharr.append(cash)
                    idarr.append(id)
            yield casharr,idarr

    def update(self, i):
        # clear plot and redraw
        self.ax.clear()
        self.ax.axis('equal')
        casharr,idarr = next(self.stream)
        self.pie = self.ax.pie(casharr, labels=idarr,autopct='%1.0f%%')
        self.ax.set_title("Timestep : {:,.0f} | Total cash : {:,.0f}".format(i,sum(casharr)))
        return self.pie,

    def save(self,filename):
        self.ani.save(filename, writer='ffmpeg', dpi=80)

    def show(self):
        plt.show()

class visualisation(object):
    def __init__(self, history_logs_list):
        self.history_logs_list = history_logs_list
        self.scatter_data = {}
        # unused data in history_logs
        #self.excess_capital = history_logs['total_excess_capital']
        #self.reinexcess_capital = history_logs['total_reinexcess_capital']
        #self.diffvar = history_logs['market_diffvar']
        #self.cumulative_bankruptcies = history_logs['cumulative_bankruptcies']
        #self.cumulative_unrecovered_claims = history_logs['cumulative_unrecovered_claims']
        return

    def insurer_pie_animation(self, run=0):
        data = self.history_logs_list[run]
        insurance_cash = np.array(data['insurance_firms_cash'])
        self.ins_pie_anim = InsuranceFirmAnimation(insurance_cash)
        return self.ins_pie_anim

    def reinsurer_pie_animation(self, run=0):
        data = self.history_logs_list[run]
        reinsurance_cash = np.array(data['reinsurance_firms_cash'])
        self.reins_pie_anim = InsuranceFirmAnimation(reinsurance_cash)
        return self.reins_pie_anim

    def insurer_time_series(self, runs=None, axlst=None, fig=None, title="Insurer", colour='black', percentiles=[25,75]):
        # runs should be a list of the indexes you want included in the ensemble for consideration
        if runs:
            data = [self.history_logs_list[x] for x in runs]
        else:
            data = self.history_logs_list
        
        # Take the element-wise means/medians of the ensemble set (axis=0)
        contracts_agg = [history_logs['total_contracts'] for history_logs in self.history_logs_list]
        profitslosses_agg = [history_logs['total_profitslosses'] for history_logs in self.history_logs_list]
        operational_agg = [history_logs['total_operational'] for history_logs in self.history_logs_list]
        cash_agg = [history_logs['total_cash'] for history_logs in self.history_logs_list]
        premium_agg = [history_logs['market_premium'] for history_logs in self.history_logs_list]

        contracts = np.mean(contracts_agg, axis=0)
        profitslosses = np.mean(profitslosses_agg, axis=0)
        operational = np.median(operational_agg, axis=0)
        cash = np.median(cash_agg, axis=0)
        premium = np.median(premium_agg, axis=0)

        self.ins_time_series = TimeSeries([
                                (contracts, 'Contracts', np.percentile(contracts_agg,percentiles[0], axis=0), np.percentile(contracts_agg, percentiles[1], axis=0)),
                                (profitslosses, 'Profitslosses', np.percentile(profitslosses_agg,percentiles[0], axis=0), np.percentile(profitslosses_agg, percentiles[1], axis=0)),
                                (operational, 'Operational', np.percentile(operational_agg,percentiles[0], axis=0), np.percentile(operational_agg, percentiles[1], axis=0)),
                                (cash, 'Cash', np.percentile(cash_agg,percentiles[0], axis=0), np.percentile(cash_agg, percentiles[1], axis=0)),
                                (premium, "Premium", np.percentile(premium_agg,percentiles[0], axis=0), np.percentile(premium_agg, percentiles[1], axis=0)),
                                        ],title=title, xlabel = "Time", axlst=axlst, fig=fig, colour=colour).plot()
        return self.ins_time_series

    def reinsurer_time_series(self, runs=None, axlst=None, fig=None, title="Reinsurer", colour='black', percentiles=[25,75]):
        # runs should be a list of the indexes you want included in the ensemble for consideration
        if runs:
            data = [self.history_logs_list[x] for x in runs]
        else:
            data = self.history_logs_list

        # Take the element-wise means/medians of the ensemble set (axis=0)
        reincontracts_agg = [history_logs['total_reincontracts'] for history_logs in self.history_logs_list]
        reinprofitslosses_agg = [history_logs['total_reinprofitslosses'] for history_logs in self.history_logs_list]
        reinoperational_agg = [history_logs['total_reinoperational'] for history_logs in self.history_logs_list]
        reincash_agg = [history_logs['total_reincash'] for history_logs in self.history_logs_list]
        catbonds_number_agg = [history_logs['total_catbondsoperational'] for history_logs in self.history_logs_list]

        reincontracts = np.mean(reincontracts_agg, axis=0)
        reinprofitslosses = np.mean(reinprofitslosses_agg, axis=0)
        reinoperational = np.median(reinoperational_agg, axis=0)
        reincash = np.median(reincash_agg, axis=0)
        catbonds_number = np.median(catbonds_number_agg, axis=0)

        self.reins_time_series = TimeSeries([
                                (reincontracts, 'Contracts', np.percentile(reincontracts_agg,percentiles[0], axis=0), np.percentile(reincontracts_agg, percentiles[1], axis=0)),
                                (reinprofitslosses, 'Profitslosses', np.percentile(reinprofitslosses_agg,percentiles[0], axis=0), np.percentile(reinprofitslosses_agg, percentiles[1], axis=0)),
                                (reinoperational, 'Operational', np.percentile(reinoperational_agg,percentiles[0], axis=0), np.percentile(reinoperational_agg, percentiles[1], axis=0)),
                                (reincash, 'Cash', np.percentile(reincash_agg,percentiles[0], axis=0), np.percentile(reincash_agg, percentiles[1], axis=0)),
                                (catbonds_number, "Activate Cat Bonds", np.percentile(catbonds_number_agg,percentiles[0], axis=0), np.percentile(catbonds_number_agg, percentiles[1], axis=0)),
                                        ],title= title, xlabel = "Time", axlst=axlst, fig=fig, colour=colour).plot()
        return self.reins_time_series

    def metaplotter_timescale(self):
        # Take the element-wise means/medians of the ensemble set (axis=0)
        contracts = np.mean([history_logs['total_contracts'] for history_logs in self.history_logs_list],axis=0)
        profitslosses = np.mean([history_logs['total_profitslosses'] for history_logs in self.history_logs_list],axis=0)
        operational = np.median([history_logs['total_operational'] for history_logs in self.history_logs_list],axis=0)
        cash = np.median([history_logs['total_cash'] for history_logs in self.history_logs_list],axis=0)
        premium = np.median([history_logs['market_premium'] for history_logs in self.history_logs_list],axis=0)
        reincontracts = np.mean([history_logs['total_reincontracts'] for history_logs in self.history_logs_list],axis=0)
        reinprofitslosses = np.mean([history_logs['total_reinprofitslosses'] for history_logs in self.history_logs_list],axis=0)
        reinoperational = np.median([history_logs['total_reinoperational'] for history_logs in self.history_logs_list],axis=0)
        reincash = np.median([history_logs['total_reincash'] for history_logs in self.history_logs_list],axis=0)
        catbonds_number = np.median([history_logs['total_catbondsoperational'] for history_logs in self.history_logs_list],axis=0)
        return

    def aux_clustered_exit_records(self, exits):
        """Auxiliary method for creation of data series on clustered events such as firm market exits.
                Will take an unclustered series and aggregate every series of non-zero elements into 
                the first element of that series.
            Arguments:
                exits: numpy ndarray or list    - unclustered series
            Returns:
                numpy ndarray of the same length as argument "exits": the clustered series."""
        exits2 = []
        ci = False
        cidx = 0
        for ee in exits:
            if ci:
                exits2.append(0)
                if ee > 0:
                    exits2[cidx] += ee
                else:
                    ci = False
            else:
                exits2.append(ee)
                if ee > 0:
                    ci = True
                    cidx = len(exits2) - 1
        
        return np.asarray(exits2, dtype=np.float64)

    def populate_scatter_data(self):
        """Method to generate data samples that do not have a time component (e.g. the size of bankruptcy events, i.e. 
                how many firms exited each time.
                The method saves these in the instance variable self.scatter_data. This variable is of type dict.
            Arguments: None.
            Returns: None."""
        
        """Record data on sizes of unrecovered_claims"""
        self.scatter_data["unrecovered_claims"] = []
        for hlog in self.history_logs_list:         # for each replication
            urc = np.diff(np.asarray(hlog["cumulative_unrecovered_claims"]))
            self.scatter_data["unrecovered_claims"] = np.hstack([self.scatter_data["unrecovered_claims"], np.extract(urc>0, urc)])
        
        """Record data on sizes of bankruptcy_events"""
        self.scatter_data["bankruptcy_events"] = []
        self.scatter_data["bankruptcy_events_relative"] = []
        self.scatter_data["bankruptcy_events_clustered"] = []
        self.scatter_data["bankruptcy_events_relative_clustered"] = []
        for hlog in self.history_logs_list:         # for each replication
            """Obtain numbers of operational firms. This is for computing the relative share of exiting firms."""
            in_op = np.asarray(hlog["total_operational"])[:-1]
            rein_op = np.asarray(hlog["total_reinoperational"])[:-1]
            op = in_op + rein_op
            
            """Obtain exits and relative exits"""
            exits = np.diff(np.asarray(hlog["cumulative_market_exits"], dtype=np.float64))
            rel_exits = exits / op
            
            """Obtain clustered exits (absolute and relative)"""
            exits2 = self.aux_clustered_exit_records(exits)            
            rel_exits2 = exits2 / op
            
            """Record data"""
            self.scatter_data["bankruptcy_events"] = np.hstack([self.scatter_data["bankruptcy_events"], np.extract(exits>0, exits)])
            self.scatter_data["bankruptcy_events_relative"] = np.hstack([self.scatter_data["bankruptcy_events_relative"], np.extract(rel_exits>0, rel_exits)])
            self.scatter_data["bankruptcy_events_clustered"] = np.hstack([self.scatter_data["bankruptcy_events_clustered"], np.extract(exits2>0, exits2)])
            self.scatter_data["bankruptcy_events_relative_clustered"] = np.hstack([self.scatter_data["bankruptcy_events_relative_clustered"], np.extract(rel_exits2>0, rel_exits2)])
            
    def show(self):
        plt.show()
        return

class compare_riskmodels(object):
    def __init__(self,vis_list, colour_list):
        # take in list of visualisation objects and call their plot methods
        self.vis_list = vis_list
        self.colour_list = colour_list
        
    def create_insurer_timeseries(self, fig=None, axlst=None, percentiles=[25,75]):
        # create the time series for each object in turn and superpose them?
        fig = axlst = None
        for vis,colour in zip(self.vis_list, self.colour_list):
            (fig, axlst) = vis.insurer_time_series(fig=fig, axlst=axlst, colour=colour, percentiles=percentiles) 

    def create_reinsurer_timeseries(self, fig=None, axlst=None, percentiles=[25,75]):
        # create the time series for each object in turn and superpose them?
        fig = axlst = None
        for vis,colour in zip(self.vis_list, self.colour_list):
            (fig, axlst) = vis.reinsurer_time_series(fig=fig, axlst=axlst, colour=colour, percentiles=percentiles) 

    def show(self):
        plt.show()
    def save(self):
        # logic to save plots
        pass

        
"""Class for CDF/cCDF distribution plots using auxiliary class from visualisation_distribution_plots.py. 
    This class arranges as many such plots stacked in one diagram as there are series in the history 
    logs they are created from, i.e. len(vis_list)."""
class CDF_distribution_plot():
    def __init__(self, vis_list, colour_list, quantiles=[.25, .75], variable="reinsurance_firms_cash", timestep=-1, plot_cCDF=True):
        """Constructor.
            Arguments:
                vis_list: list of visualisation objects - objects hilding the data
                colour list: list of str                - colors to be used for each plot
                quantiles: list of float of length 2    - lower and upper quantile for inter quantile range in plot
                variable: string (must be a valid dict key in vis_list[i].history_logs_list
                                                        - the history log variable for which the distribution is plotted
                                                            (will be either "insurance_firms_cash" or "reinsurance_firms_cash")
                timestep: int                           - timestep at which the distribution to be plotted is taken
                plot_cCDF: bool                         - plot survival function (cCDF) instead of CDF
            Returns class instance."""
        self.vis_list = vis_list
        self.colour_list = colour_list
        self.lower_quantile, self.upper_quantile = quantiles
        self.variable = variable
        self.timestep = timestep
    
    def generate_plot(self, xlabel=None, filename=None):
        """Method to generate and save the plot.
            Arguments:
                xlabel: str or None     - the x axis label
                filename: str or None   - the filename without ending
            Returns None."""

        """Set x axis label and filename to default if not provided"""
        xlabel = xlabel if xlabel is not None else self.variable
        filename = filename if filename is not None else "CDF_plot_" + self.variable
        
        """Create figure with correct number of subplots"""
        self.fig, self.ax = plt.subplots(nrows=len(self.vis_list))
        
        """Loop through simulation record series, populate subplot by subplot"""
        for i in range(len(self.vis_list)):
            """Extract firm records from history logs"""
            series_x = [replication[self.variable][self.timestep] for replication in self.vis_list[i].history_logs_list]
            """Extract the capital holdings from the tuple"""
            for j in range(len(series_x)):
                series_x[j] = [firm[0] for firm in series_x[j] if firm[2]]
            """Create CDFDistribution object and populate the subfigure using it"""
            VDP = visualisation_distribution_plots.CDFDistribution(series_x)
            #VDP.make_figure(upper_quantile=self.upper_quantile, lower_quantile=self.lower_quantile) 
            c_xlabel = "" if i < len(self.vis_list) - 1 else xlabel
            VDP.plot(ax=self.ax[i], ylabel="cCDF " + str(i+1) + "RM", xlabel=c_xlabel, upper_quantile=self.upper_quantile, lower_quantile=self.lower_quantile, color=self.colour_list[i], plot_cCDF=True)
        
        """Finish and save figure"""
        self.fig.tight_layout()
        self.fig.savefig(filename + ".pdf")
        self.fig.savefig(filename + ".png", density=300)


"""Class for CDF/cCDF distribution plots using auxiliary class from visualisation_distribution_plots.py. 
    This class arranges as many such plots stacked in one diagram as there are series in the history 
    logs they are created from, i.e. len(vis_list)."""
class Histogram_plot():
    def __init__(self, vis_list, colour_list, variable="bankruptcy_events"):
        """Constructor.
            Arguments:
                vis_list: list of visualisation objects - objects hilding the data
                colour list: list of str                - colors to be used for each plot
                variable: string (must be a valid dict key in vis_list[i].scatter_data
                                                        - the history log variable for which the distribution is plotted
            Returns class instance."""
        self.vis_list = vis_list
        self.colour_list = colour_list
        self.variable = variable
    
    def generate_plot(self, xlabel=None, filename=None, logscale=False):
        """Method to generate and save the plot.
            Arguments:
                xlabel: str or None     - the x axis label
                filename: str or None   - the filename without ending
            Returns None."""

        """Set x axis label and filename to default if not provided"""
        xlabel = xlabel if xlabel is not None else self.variable
        filename = filename if filename is not None else "Histogram_plot_" + self.variable
        
        """Create figure with correct number of subplots"""
        self.fig, self.ax = plt.subplots(nrows=len(self.vis_list))
        
        #pdb.set_trace()

        """find max and min values"""
        """combine all data sets"""
        all_data = [np.asarray(vl.scatter_data[self.variable]) for vl in self.vis_list]
        all_data = np.hstack(all_data)
        
        """Catch empty data sets"""
        if len(all_data) == 0:
            return    
        #all_data = []
        #for vl in self.vis_list:
        #    for item in vl.scatter_data[self.variable]:
        #        all_data += item
        minmax = (np.min(all_data), np.max(all_data))
        num_bins = min(25, len(np.unique(all_data)))
        
        """Loop through simulation record series, populate subplot by subplot"""
        for i in range(len(self.vis_list)):
            """Extract records from history logs"""
            scatter_data = self.vis_list[i].scatter_data[self.variable]
            """Create Histogram object and populate the subfigure using it"""
            H = visualisation_distribution_plots.Histogram(scatter_data)
            c_xlabel = "" if i < len(self.vis_list) - 1 else xlabel
            H.plot(ax=self.ax[i], ylabel="Dens. " + str(i+1) + "RM", xlabel=c_xlabel, color=self.colour_list[i], num_bins=num_bins, logscale=logscale, xlims=minmax)
            
        """Finish and save figure"""
        self.fig.tight_layout()
        self.fig.savefig(filename + ".pdf")
        self.fig.savefig(filename + ".png", density=300)
    
if __name__ == "__main__":


    # use argparse to handle command line arguments
    parser = argparse.ArgumentParser(description='Model the Insurance sector')
    parser.add_argument("--single", action="store_true", help="plot time series of a single run of the insurance model")
    parser.add_argument("--comparison", action="store_true", help="plot the result of an ensemble of replicatons of the insurance model")
    parser.add_argument("--firmdistribution", action="store_true", help="plot the cCDFs of firm size distributions with quantiles indicating variation across ensemble")
    parser.add_argument("--bankruptcydistribution", action="store_true", help="plot the histograms of bankruptcy events across ensemble")

    args = parser.parse_args()


    if args.single:

        # load in data from the history_logs dictionarywith open("data/history_logs.dat","r") as rfile:
        with open("data/history_logs.dat","r") as rfile:
            history_logs_list = [eval(k) for k in rfile] # one dict on each line
        # first create visualisation object, then create graph/animation objects as necessary
        vis = visualisation(history_logs_list)
        vis.insurer_pie_animation()
        vis.reinsurer_pie_animation()
        vis.insurer_time_series()
        vis.reinsurer_time_series()
        vis.show()
        N = len(history_logs_list)


    if args.comparison or args.firmdistribution or args.bankruptcydistribution:
    
        # for each run, generate an animation and time series for insurer and reinsurer
        # TODO: provide some way for these to be lined up nicely rather than having to manually arrange screen
        #for i in range(N):
        #    vis.insurer_pie_animation(run=i)
        #    vis.insurer_time_series(runs=[i])
        #    vis.reinsurer_pie_animation(run=i)
        #    vis.reinsurer_time_series(runs=[i])
        #    vis.show()
        vis_list = []
        filenames = ["./data/" + x + "_history_logs.dat" for x in ["one","two","three","four"]]
        for filename in filenames:
            with open(filename,'r') as rfile:
                history_logs_list = [eval(k) for k in rfile] # one dict on each line
                vis_list.append(visualisation(history_logs_list))

        colour_list = ['red', 'blue', 'green', 'yellow']
    #pdb.set_trace()
            
    if args.comparison:
        
        cmp_rsk = compare_riskmodels(vis_list, colour_list)
        cmp_rsk.create_insurer_timeseries(percentiles=[10,90])
        cmp_rsk.create_reinsurer_timeseries(percentiles=[10,90])
        cmp_rsk.show()
    
    if args.firmdistribution:
        CP = CDF_distribution_plot(vis_list, colour_list, variable="insurance_firms_cash", timestep=-1, plot_cCDF=True)  
        CP.generate_plot(xlabel="Firm size (capital)")
        if not isleconfig.simulation_parameters["reinsurance_off"]:
            CP = CDF_distribution_plot(vis_list, colour_list, variable="reinsurance_firms_cash", timestep=-1, plot_cCDF=True)  
            CP.generate_plot(xlabel="Firm size (capital)")
    
    if args.bankruptcydistribution:
        for vis in vis_list:
            vis.populate_scatter_data()
        HP = Histogram_plot(vis_list, colour_list, variable="bankruptcy_events")  
        HP.generate_plot(logscale=True, xlabel="Number of bankruptcies")
        HP = Histogram_plot(vis_list, colour_list, variable="bankruptcy_events_relative")  
        HP.generate_plot(logscale=True, xlabel="Share of bankrupt firms")
        HP = Histogram_plot(vis_list, colour_list, variable="bankruptcy_events_clustered")  
        HP.generate_plot(logscale=True, xlabel="Number of bankruptcies")
        HP = Histogram_plot(vis_list, colour_list, variable="bankruptcy_events_relative_clustered")  
        HP.generate_plot(logscale=True, xlabel="Share of bankrupt firms")
        HP = Histogram_plot(vis_list, colour_list, variable="unrecovered_claims")  
        HP.generate_plot(logscale=True, xlabel="Damages not recovered")
