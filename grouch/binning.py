import numpy as np
from scipy.fft import fft,ifft
from copy import copy
import numba as nb

@nb.njit('(float64[:,::1],int64[::1],float64[::1,:])')
def add_at(a,indices,b):
    """Faster version of numpy.add.at (about x50 faster.)
    
    NOTE: the signatures for a and b look different because use-case is for
    a created as a numpy array with C-contiguous order, whereas 
    pandas.DataFrame.to_numpy() for some reason creates an array with 
    fortran-contiguous order. 

    Args:
        a (ndarray, np.float32): Array to add to. 
        indices (ndarray, np.int64): Indices to add at.
        b (ndarray, np.float32): Array for adding.
    """
    for i in range(indices.shape[0]):
        for j in range(b.shape[1]):
            a[indices[i],j] += b[i,j]
            
@nb.njit('(int64[:,::1],int64[::1],boolean[::1,:])')
def addBool_at(a,indices,b):
    """Faster version of numpy.add.at (about x50 faster.)
    
    NOTE: the signatures for a and b look different because use-case is for
    a created as a numpy array with C-contiguous order, whereas 
    pandas.DataFrame.to_numpy() for some reason creates an array with 
    fortran-contiguous order. 

    Args:
        a (ndarray, np.float32): Array to add to.
        indices (ndarray, np.int64): Indices to add at.
        b (ndarray, np.bool_): Array for adding.
    """
    for i in range(indices.shape[0]):
        for j in range(b.shape[1]):
            a[indices[i],j] += b[i,j]

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

def process_dataframe(dataframe,*data_bins,scan_section="hatch",**kwargs):
    """Take a dataframe, a sequence of DataBins objects, 
    and carry out binning for each object on the dataframe.

    Args:
        dataframe (pd.DataFrame): Pandas dataframe
        scan_section (string, optional): Which section of scan paths to use. Options are 'hatch', 'perimeter', 'all'. Defaults to True.

    Returns:
        tuple,DataBins: binned databins objects. 
    """
    hrst = kwargs.get("hatch_reconstruction_score_threshold",0.9)
    if not scan_section in ("hatch","perimeter","all"):
        raise ValueError("scan_section argument must be one of 'hatch', 'perimeter' or 'all'.")
    
    starts,ends = conditional_breakpoints(dataframe,**kwargs)
    for start,end in zip(starts,ends):
        if end - start <= 1:
            # This is a bit of a hack and should probably be accounted for in the 
            # function conditional_breakpoints
            continue
        if scan_section=='hatch' or scan_section=='perimeter':
            x,y = dataframe.loc[start:end,("X","Y")].values.T
            # Determine whether this segment corresponds to hatching or some other part of the build layer. 
            score = fft_recon_score(x,y,**kwargs)
            if scan_section == 'hatch' and score > hrst:
                continue
            elif scan_section == 'perimeter' and score <= hrst:
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
        self.sum_bins   = np.zeros((np.prod(self.nbin),
                                    len(variable_names),),
                             dtype=np.float64)
        self.sum2_bins  = self.sum_bins.copy()
        self.count_bins = self.sum_bins.copy().astype(np.int64)
        
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
        values = dataframe.loc[:,self.variable_names].fillna(0.0).to_numpy(dtype=np.float64)
        finite_vals = np.isfinite(dataframe.loc[:,self.variable_names]).to_numpy(dtype=np.bool_)
        add_at(self.sum_bins,xy,values)
        add_at(self.sum2_bins,xy,values**2)
        addBool_at(self.count_bins,xy,finite_vals)
        
    def __getitem__(self,indices):
        if isinstance(indices,tuple):
            # If a tuple, it should have at the same length as the number of coordinates
            if len(indices) < self.D:
                raise ValueError("Trying to slice along too few dimensions ({}). This object has {} coordinates.".format(len(tuple),self.D))
            elif len(indices) > self.D:
                raise ValueError("Trying to slice along too many dimensions ({}). This object only has {} coordinates.".format(len(tuple),self.D))
        else:
            indices = (indices,) # For ease of handling 
        # 'slice' this object's attributes
        new_coord_names = []
        new_bin_edges = []
        bin_indices = []
        for idx,coord_name,edges in zip(indices,self.coord_names,self.bin_edges):
            if isinstance(idx,slice):
                if idx.step is not None:
                    if abs(idx.step) != 1:
                       raise ValueError("Slicing only supports continuous slicing.")
                start,stop,step = idx.indices(len(edges)-1)
                bin_indices.append(list(range(*slice(start+1,stop+1,step).indices(len(edges)+1)))) # +1 accounts for the extra bin at each end
                # Now modify to account for the next step where we get edges not bins. 
                stop +=1 if step==1 else stop
                start +=1 if idx.step==-1 else start
                new_coord_names.append(coord_name)
                new_bin_edges.append(edges[slice(start,stop,idx.step)])
            elif isinstance(idx,int):
                bin_indices.append([idx+1])
            else:
                raise ValueError("Indices must be supplied as slice or int.")
        
        new_variable_names = copy(self.variable_names) # Keep all of the variables. 
        new_object = DataBins(new_coord_names,new_bin_edges,new_variable_names,**self.kwargs)
        # Can now slice this object's bins
        old_bin_indices = np.ravel_multi_index(tuple(bin_indices),self.nbin)
        new_bin_indices = np.ravel_multi_index([np.arange(1,nbin-1) for nbin in new_object.nbin],new_object.nbin) # Don't care about the extra bins at each end
        new_object.sum_bins[new_bin_indices,...] = self.sum_bins[old_bin_indices,...]
        new_object.sum2_bins[new_bin_indices,...] = self.sum2_bins[old_bin_indices,...]
        new_object.count_bins[new_bin_indices,...] = self.count_bins[old_bin_indices,...]
        
        return new_object
    
    def _calc_mean_per_bin(self,index):
        return np.divide(self.sum_bins[:,index],self.count_bins[:,index],out=np.zeros_like(self.sum_bins[:,index]),where=self.count_bins[:,index]>0)
    
    def _calc_std_per_bin(self,index):
        return np.sqrt(
            np.divide(self.sum2_bins[:,index],self.count_bins[:,index],out=np.zeros_like(self.sum2_bins[:,index]),where=self.count_bins[:,index]>0) -\
                np.divide(self.sum_bins[:,index],self.count_bins[:,index],out=np.zeros_like(self.sums[:,index]),where=self.count_bins[:,index]>0) **2 )
        
    def _calc_count_per_bin(self,index):
        return self.count_bins[:,index]
    
    def _calc_avg_per_distance_traversed_per_bin(self,index):
        return np.divide(self.sum_bins[:,index],self.sum_bins[:,self.variable_names.get_loc("Distance traversed")],
                         out=np.zeros_like(self.sum_bins[:,index]),where=self.count_bins[:,index]>0)
    
    def calculate_per_bin(self,quantity,variable):
        """Calculate a given quantity (mean, std, etc.) over a given variable, on a per-bin basis.
        
        Args:
            quantity (string or callable): a string chosen from "mean", "std", "count", "per_traversed" or else a callable that accepts DataBins, index as inputs.
            variable (string): a variable to calculate the supplied quantity for. 
            
        Return:
            ndarray: represents the requested calculations for each bin.  
        """
        # Error messages
        err_0 = "DataBins should contain data for variable = '{}'".format(variable)
        err_1 = "DataBins should contain data for variable = 'Distance traversed' in order to use arg quantity = 'per_traversed'"
        
        # Check if variables exist
        if not variable in self.variable_names:
            raise ValueError(err_0)
        elif not "Distance traversed" in self.variable_names and quantity == "per_traversed":
            raise ValueError(err_1)
        
        if callable(quantity):
            func_ = quantity
        elif quantity == "mean":
            func_ = self._calc_mean_per_bin
        elif quantity == "std":
            func_ = self._calc_std_per_bin
        elif quantity == "count":
            func_ = self._calc_count_per_bin
        elif quantity == "per_traversed":
            func_ = self._calc_avg_per_distance_traversed_per_bin
        else:
            raise ValueError("quantity arg should be 'mean', 'std', 'count', 'per_traversed', or a callable.")
        return func_(self.variable_names.get_loc(variable)).reshape(*self.nbin)
        
        
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
    theta = np.arctan2(v[:,1],v[:,0])
    # Acceleration
    v_mid = r[1:,:] - r[:-1,:]
    a = v_mid[1:,:] - v_mid[:-1,:]
    a = np.r_[[v_mid[0,:]],a,[v_mid[-1,:]]]
    accel = (a*v).sum(axis=1) / np.linalg.norm(v)
    # Add columns
    df["Orientation"] = theta
    df["Speed"] = np.linalg.norm(v,axis=1)
    df["Acceleration"] = accel
    df["Distance traversed"] = np.r_[0.0,np.linalg.norm(v_mid,axis=1)]
    return df