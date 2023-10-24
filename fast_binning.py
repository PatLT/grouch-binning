import numpy as np
import pandas as pd 
from scipy.fft import fft,ifft
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

def adjust_and_pad(x,y,pad=1.0):
    """Apply some normalisation, then padding to spatial path data,
    ready for FFT application. 

    Args:
        x (ndarray): Path's x coords
        y (ndarray): Path's y coords
        pad (float, optional): Fraction to pad array by ~on each side~. Defaults to 1.0.

    Returns:
        ndarray,complex128 : Complex representation of the padded and normalised path.
    """
    x_min = x.min() ; x_max = x.max()
    y_min = y.min() ; y_max = y.max()
    x_mid = 0.5*(x_min+x_max)
    y_mid = 0.5*(y_min+y_max)
    r_max = np.sqrt((x-x_mid)**2 + (y-y_mid)**2).max()
    if r_max < 1.e-8:
        return None
    if pad > 0.0:
        out = np.zeros(int(np.ceil(((2*pad+1)*x.shape[0]))),dtype=np.cdouble)
        start = int(np.ceil(pad*x.shape[0]))
        end = int(np.ceil(pad*x.shape[0]))+x.shape[0]
        out[start:end] = ((x-x_mid)+(y-y_mid)*1j) / r_max
    else:
        out = ((x - x_mid) + (y - y_mid)*1j)/r_max
    return out

def unpad(z,pad):
    """Undo padding but NOT normalisation. 
    For use after applying filtering and IFFT

    Args:
        z (ndarray): Complex path.
        pad (float): padding fraction to remove. 

    Returns:
        ndarray: Complex path without padding. 
    """
    if pad > 0.0:
        old_shape = int(np.floor(z.shape[0]/(2*pad+1)))
        start = int(np.ceil(pad*old_shape))
        end = int(np.ceil(pad*old_shape))+old_shape
        return z[start:end]
    else:
        return z
    
def fft_recon_score(x,y,**kwargs):
    """Take a path in the XY plane, calculate its fourier transform.
    Then filter out all but the smallest frequencies and reconstruct the path using
    IFFT. Calculate an L2 metric between the reconstructed path and the original path 
    and use this to calculate a score - should be worse for hatching paths than for 
    e.g. perimeter paths. 
    
    $score = 1.0 - \sum((x,y)-(x_{LF},y_{LF}))^2$

    Args:
        x (ndarray): Path's x coords 
        y (ndarray): Path's y coords

    Returns:
        float: Reconstruction score (closer to 1.0 is better)
    """
    pad = kwargs.get("fft_pad",1.0)
    f_c_factor = kwargs.get("fft_freq_cutoff_factor",2.0)
    # Pad x,y 
    z = adjust_and_pad(x,y,pad)
    if z is None:
        return 0.0
    R = fft(z)
    freq_cutoff = f_c_factor * 2*np.pi/(z.shape[0]/(1+2*pad))
    ind_cutoff = int(np.ceil(freq_cutoff * z.shape[0]))
    # Mask R based on cutoff
    R_masked = np.zeros_like(R)
    R_masked[:ind_cutoff] = R[:ind_cutoff]
    R_masked[-ind_cutoff:] = R[-ind_cutoff:]
    z_tr = ifft(R_masked)
    # Score 
    score = 1.0 - np.abs(unpad(z,pad) - unpad(z_tr,pad)).mean()
    return score

def conditional_breakpoints(dataframe,**kwargs):
    """Break a dataframe into sections using a condition. 

    Args:
        dataframe (pd.DataFrame): A pandas dataframe. 

    Returns:
        ndarray,ndarray: Start and end points for each section. 
    """
    status_threshold = kwargs.get("status_threshold",0.0)
    status_variable =  kwargs.get("status_variable","P")
    # Determine locations where machine status switches
    on_status = dataframe.loc[:,status_variable].values > status_threshold
    ends   = np.argwhere(on_status[:-2] * (~on_status)[1:-1] * (~on_status)[2:]).ravel()
    starts = np.argwhere(on_status[2:]  * (~on_status)[1:-1]  * (~on_status)[:-2]).ravel() + 2
    if on_status[0]:
        starts = np.r_[[0],starts]
    if on_status[-1]:
        ends = np.r_[ends,[on_status.shape[0]]]
    return starts,ends

def process_dataframe(dataframe,*data_bins,hatch_only=True,**kwargs):
    """Take a dataframe, a sequence of DataBins objects, 
    and carry out binning for each object on the dataframe.

    Args:
        dataframe (pd.DataFrame): Pandas dataframe
        hatch_only (bool, optional): Whether to only bin data for hatch scans. Defaults to True.

    Returns:
        tuple,DataBins: binned databins objects. 
    """
    hrst = kwargs.get("hatch_reconstruction_score_threshold",0.9)
    
    starts,ends = conditional_breakpoints(dataframe,**kwargs)
    for start,end in zip(starts,ends):
        if hatch_only:
            x,y = dataframe.loc[start:end,("X","Y")].values.T
            # Determine whether this segment corresponds to hatching or some other part of the build layer. 
            score = fft_recon_score(x,y,**kwargs)
            if score > hrst:
                continue
        # Assuming the section is valid
        for db in data_bins:
            db.bin(dataframe.loc[start:end,:])
            
    return data_bins

def optimal_grid(coord_names,dataframe,init_grid_shape,sub_divisions=1,aux_variable=None,tol=10.0):
    """
        Semi-dynamically generate optimal bin_coords from a dataset. For now the initial grid has to be regular 
        (init_grid_shape merely specifies the initial number of subdivisions along each coordinate axis.)

        Args:
            coord_names (list): Unique labels for each dependent variable i.e. coordinate.
            sample_dataframe (pd.DataFrame): A dataframe, the data from which will be used to determine the binning. 
            init_grid_shape (tuple): Number of subdivisions along each coordinate axis. 
            sub_divisions (int, optional): Specify the number of subdivisions within the dynamically chosen bins. Defaults to 1.
            aux_variable (string, optional): Specify another column (variable) from the dataframe that will be used for filtering. Defaults to None.
            tol (float, optional): An aditional padding or tolerance that's added around data locations to give the final bins. Defaults to 10.0.
    """
    grid = []
    for num_,coord in zip(init_grid_shape,coord_names):
        sub_grid = []
        x = dataframe.loc[:,[coord]].where(dataframe[aux_variable] > 0,inplace=False).values if aux_variable is not None else dataframe.loc[:,[coord]].values
        x_ = x[np.isfinite(x)]
        init_sub_grid = np.linspace(x_.min()-tol,x_.max()+tol,num_+1,endpoint=True)
        bin_assignments = np.searchsorted(init_sub_grid,x_,side='left') - 1
        # Loop over all the bins on this axis, and determine the corresponding sub-grid. 
        for i in range(num_):
            if i in bin_assignments:
                bin0_l = max(x_[bin_assignments==i].min() - tol,init_sub_grid[i])
                bin0_u = min(x_[bin_assignments==i].max() + tol,init_sub_grid[i+1])
                # Check for overlap with current subgrid
                if i>0:
                    if bin0_l <= sub_grid[-1]:
                        sub_grid = np.r_[sub_grid,np.linspace(bin0_l,bin0_u,sub_divisions+1,endpoint=True)[1:]]
                        continue
                # Otherwise, (no overlap)
                sub_grid = np.r_[sub_grid,np.linspace(bin0_l,bin0_u,sub_divisions+1,endpoint=True)]
            else:
                sub_grid = np.r_[sub_grid,[init_sub_grid[i],init_sub_grid[i+1]]]
        grid.append(sub_grid)
    return grid
        
class DataBins():
    def __init__(self,coord_names,bin_edges,variable_names,**kwargs) -> None:
        """This object represents how data is binned over a given variable. 

        Args:
            coord_names (list of strings): Coordinate names: to be sourced directly from the dataframe columns.
            bin_edges (list): A sequence of arrays describing the monotonically increasing bin edges along each dimension
            variable_names (list of strings): Variables to bin into the data bins. 
        """
        self.coord_names = coord_names
        self.bin_edges = bin_edges
        self.variable_names = variable_names
        self.D = len(coord_names)
        self.nbin = [len(edges)+1 for edges in self.bin_edges]# +1 accounts for outliers
        
        self.kwargs = kwargs
        
        # Flattened bins
        # We include two extra bins to account for outliers. 
        self.sum_bins   = np.zeros((np.prod([len(dim_bin_edges)+1 for dim_bin_edges in self.bin_edges]),
                                    len(variable_names),),
                             dtype=np.single)
        self.sum2_bins = self.sum_bins.copy()
        self.count_bins = self.sum_bins.copy().astype(np.uint32)
        
    def bin(self,dataframe):
        """Efficient datatframe binning. 
        
        Code based on numpy.histogramdd: https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/histograms.py#L901-L1072

        Args:
            dataframe (pd.DataFrame): Dataframe to perform binning on.
        """
        # Compute bin numbers along each axis for each datapoint
        Ncount = tuple(np.searchsorted(self.bin_edges[i],dataframe.loc[:,coord_name].values,side='right') for i,coord_name in enumerate(self.coord_names))
        for i,coord_name in enumerate(self.coord_names):
            # Check for points on the rightmost bin edge
            on_edge = dataframe.loc[:,coord_name].values == self.bin_edges[i][-1]
            # Move these points one bin to the left
            Ncount[i][on_edge] -= 1
        # Use ravel_multi_index to calculate actual bin numbers
        xy = np.ravel_multi_index(Ncount,self.nbin)
        #hist = np.bincount(xy,minlength=np.prod(self.nbin))
        # It's possible that using bincount is faster - need to check. 
        np.add.at(self.sum_bins,xy[None,:],dataframe.loc[:,self.variable_names].values)
        np.add.at(self.sum2_bins,xy[None,:],dataframe.loc[:,self.variable_names].values**2)
        np.add.at(self.count_bins,xy[None,:],np.isfinite(dataframe.loc[:,self.variable_names]))
        
def expand_df(df):
    """Expand a dataframe with extra variables:
    * orientation: converts displacement vector into an orientation angle.
    * speed: calculates a velocity vectors and norms it. 
    * acceleration: similar to speed. 

    Args:
        df (pd.DataFrame): Dataframe to expand

    Returns:
        pd.DataFrame: expanded dataframe.
    """
    r = df.loc[:,("X","Y")].values
    # Orientation
    v = r[2:,:] - r[:-2,:]
    v = np.r_[[r[1,:]-r[0,:]],v,[r[-1,:]-r[-2,:]]]
    theta = np.arctan2(v[:,0],v[:,1])
    # Acceleration
    v_mid = r[1:,:] - r[:-1,:]
    a = v_mid[1:,:] - v_mid[:-1,:]
    a = np.r_[[v_mid[0,:]],a,[v_mid[-1,:]]]
    accel = (a*v).sum(axis=1) / np.linalg.norm(v)
    # Add columns
    df["Orientation"] = theta
    df["Speed"] = np.linalg.norm(v,axis=1)
    df["Acceleration"] = accel
    return df

def calc_plot_range(*data_vals,non_zero=False):
    min_ = np.inf
    max_ = -np.inf
    for values in data_vals:
        new_min = np.ma.masked_equal(values,0.0,copy=True).min() if non_zero else values.min()
        new_max = values.max()
        max_ = new_max if new_max > max_ else max_
        min_ = new_min if new_min < min_ else min_
    oom = np.floor(np.log10(max_ - min_))
    return 10**oom * np.floor(min_/10**oom), 10**oom * np.ceil(max_/10**oom)

def calc_colour_range(*data_vals,non_zero=True):
    min_ = np.inf
    max_ = -np.inf
    for values in data_vals:
        new_min = np.ma.masked_equal(values,0.0,copy=True).min() if non_zero else values.min()
        new_max = values.max()
        max_ = new_max if new_max > max_ else max_
        min_ = new_min if new_min < min_ else min_
    return min_,max_

def plot_bins(fig,variable,xy_plane_hist=None,theta_hist=None,z_hist=None,quantity="mean"):
    # Function used to calculate the plot quantity. 
    if callable(quantity):
        func_ = quantity
    elif quantity == "mean":
        func_ = lambda histogram,index: np.divide(histogram.sum_bins[:,index],histogram.count_bins[:,index],out=np.zeros_like(histogram.sum_bins[:,index]),where=histogram.count_bins[:,index]>0)
    elif quantity == "std":
        func_ = lambda histogram,index: np.sqrt(
            np.divide(histogram.sum2_bins[:,index],histogram.count_bins[:,index],out=np.zeros_like(histogram.sum2_bins[:,index]),where=histogram.count_bins[:,index]>0) -\
                np.divide(histogram.sum_bins[:,index],histogram.count_bins[:,index],out=np.zeros_like(histogram.sums[:,index]),where=histogram.count_bins[:,index]>0) **2 )
    elif quantity == "count":
        func_ = lambda histogram,index: histogram.count_bins[:,index]
    else:
        raise ValueError("quantity arg should be 'mean', 'std', 'count' or a callable.")
    
    num_axes = sum([0 if hist is None else w for hist,w in zip((xy_plane_hist,theta_hist,z_hist),(1,2,2))])
    ax_ind = 1
    
    # Z-histogram
    if z_hist is not None:
        ax_z = fig.add_subplot(1,num_axes,ax_ind)
        index = z_hist.variable_names.get_loc(variable)
        values = func_(z_hist,index)[1:-1]
        # Make plot
        ax_z.barh(0.5*(z_hist.bin_edges[0][1:] + z_hist.bin_edges[0][:-1]),
            values,
            z_hist.bin_edges[0][1:] - z_hist.bin_edges[0][:-1],
            color='orange'
        )
        ax_z.set_ylabel("layer number")
        ax_z.set_xlim(*calc_plot_range(values))
        ax_z.set_ylim(z_hist.bin_edges[0][0],z_hist.bin_edges[0][-1])
        
        ax_ind += 1
    else:
        ax_z = None
    
    # XY planar heatmap
    if xy_plane_hist is not None:
        ax_xy = fig.add_subplot(1,num_axes,(ax_ind,ax_ind+1))
        index = xy_plane_hist.variable_names.get_loc(variable)
        values = func_(xy_plane_hist,index).reshape(xy_plane_hist.nbin[0],xy_plane_hist.nbin[1])[1:-1,1:-1]
        # Colormapping
        cmap = plt.colormaps["afmhot_r"]
        vmin,vmax = calc_colour_range(values,non_zero=True)
        # Make plot 
        for i,(x_l,x_u) in enumerate(zip(xy_plane_hist.bin_edges[0][:-1],(xy_plane_hist.bin_edges[0][1:]))):
            for j,(y_l,y_u) in enumerate(zip(xy_plane_hist.bin_edges[0][:-1],(xy_plane_hist.bin_edges[0][1:]))):
                v = values[i,j]
                ax_xy.fill_between([x_l,x_u],[y_l,y_l],[y_u,y_u],color=cmap(v/vmax))
        ax_xy.set_xlabel("x coordinate")
        ax_xy.set_ylabel("y coordinate")
        
        ax_ind += 2
    else:
        ax_xy = None
    
    # Orientation histogram
    if theta_hist is not None:
        # Some plotting parameters
        bot = 30.0
        top = 100.0
        
        ax_th = fig.add_subplot(1,num_axes,(ax_ind,ax_ind+1),polar=True)
        index = theta_hist.variable_names.get_loc(variable)
        values = func_(theta_hist,index)[1:-1]
        min_,max_ = calc_plot_range(values)
        # Make plot 
        ax_th.bar(0.5*(theta_hist.bin_edges[0][1:] + theta_hist.bin_edges[0][:-1]),
            bot + (top-bot) * (values - min_)/(max_ - min_),
            width = theta_hist.bin_edges[0][1:] - theta_hist.bin_edges[0][:-1],
            bottom = bot,
            color='orange'
        )
        ax_th.set_rticks([bot,0.5*(bot+top),top],labels=[min_,0.5*(min_+max_),max_])
        
        ax_ind += 2
    else:
        ax_th = None
    
    return ax_z,ax_xy,ax_th