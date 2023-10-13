import numpy as np
import pandas as pd 
from scipy.fft import fft,ifft

def adjust_and_pad(x,y,pad=1.0):
    x_min = x.min() ; x_max = x.max()
    y_min = y.min() ; y_max = y.max()
    x_mid = 0.5*(x_min+x_max)
    y_mid = 0.5*(y_min+y_max)
    r_max = np.sqrt((x-x_mid)**2 + (y-y_mid)**2).max()
    if pad > 0.0:
        out = np.zeros(int(np.ceil(((2*pad+1)*x.shape[0]))),dtype=np.cdouble)
        start = int(np.ceil(pad*x.shape[0]))
        end = int(np.ceil(pad*x.shape[0]))+x.shape[0]
        out[start:end] = ((x-x_mid)+(y-y_mid)*1j) / r_max
    else:
        out = ((x - x_mid) + (y - y_mid)*1j)/r_max
    return out

def unpad(z,pad):
    if pad > 0.0:
        old_shape = int(np.floor(z.shape[0]/(2*pad+1)))
        start = int(np.ceil(pad*old_shape))
        end = int(np.ceil(pad*old_shape))+old_shape
        return z[start:end]
    else:
        return z
    
def fft_recon_score(x,y,**kwargs):
    pad = kwargs.get("fft_pad",1.0)
    f_c_factor = kwargs.get("fft_freq_cutoff_factor",2.0)
    # Pad x,y 
    z = adjust_and_pad(x,y,pad)
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
    hrst = kwargs.get("hatch_reconstruction_score_threshold",0.9)
    
    starts,ends = conditional_breakpoints(dataframe,kwargs)
    for start,end in zip(starts,ends):
        if hatch_only:
            x,y = dataframe.loc[start:end,("X","Y")].values.T
            # Determine whether this segment corresponds to hatching or some other part of the build layer. 
            score = fft_recon_score(x,y,kwargs)
            if score < hrst:
                continue
        # Assuming the section is valid
        for db in data_bins:
            db.bin(dataframe.loc[start:end,:])
            
    return data_bins
        
class DataBins():
    def __init__(self,coord_names,bin_edges,variable_names,**kwargs) -> None:
        self.coord_names = coord_names
        self.bin_edges = bin_edges
        self.variable_names = variable_names
        self.D = len(coord_names)
        self.nbin = [len(edges)+1 for edges in self.bin_edges]# +1 accounts for outliers
        
        self.kwargs = kwargs
        
        # bins
        self.count_bins = np.zeros((len(dim_bin_edges)-1 for dim_bin_edges in self.bin_edges),
                             dtype=np.uint)
        self.sum_bins   = np.zeros((len(dim_bin_edges)-1 for dim_bin_edges in self.bin_edges) \
                                    + (len(variable_names),),
                             dtype=np.single)
        self.sum2_bins = self.sum_bins.copy()
        
    def bin(self,dataframe):
        # Compute bin numbers along each axis for each datapoint
        Ncount = (np.searchsorted(self.edges[i],dataframe.loc[:,coord_name].values) for i,coord_name in enumerate(self.coord_names))
        for i,coord_name in enumerate(self.coord_names):
            # Check for points on the rightmost bin edge
            on_edge = (self.edges[i],dataframe.loc[:,coord_name].values == self.edges[i][-1])
            # Move these points one bin to the left
            Ncount[i][on_edge] -= 1
        # Use ravel_multi_index to calculate actual bin numbers
        xy = np.ravel_multi_index(Ncount,self.nbin)
        
        
