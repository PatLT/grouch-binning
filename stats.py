import numpy as np
import pickle

class Histogram():
    def __init__(self,coord_names,variable_names,*bin_coords,aux_variable = None) -> None:
        """
        A class to hold the bins that store summary statistics on each independent variable. Said independent variables are labelled by variable_names. 

        Args:
            coord_names (list): Unique labels for each dependent variable i.e. coordinate.
            variable_names (list): Unique labels for each independent variable. 
            bin_coords (ndarray): Coordinates of the bin EDGES. 
            aux_variable (string, optional): Specify another column (variable) from the dataframe that will be used for filtering. Defaults to None.
        """
        self.bin_coords = bin_coords
        self.num_bins = np.prod([len(coords)-1 for coords in bin_coords])    
        self.variable_names = variable_names
        self.coord_names = coord_names
        self.aux_variable = aux_variable
        # Things to assign later
        self.counts = np.zeros((len(self.variable_names),self.num_bins),dtype=int)
        self.sums   = np.zeros((len(self.variable_names),self.num_bins))
        self.sum2   = np.zeros((len(self.variable_names),self.num_bins))
        
    def pickle(self,filename=None,suffix=""):
        """Pickle the histogram.

        Args:
            filename (string, optional): Filename to pickle to. Defaults to None.
            suffix (strong, optional): An extra description suffix for this file. 
        """
        filename = "HIST_"+"_".join(self.coord_names) + suffix + (str(len(self.variable_names)) if len(self.variable_names) > 1 else self.variable_names[0]) if filename is None else filename.split(".")[0]
        filename += ".pkl"
        with open(filename,"wb") as f:
            pickle.dump(self,f)
            
    @classmethod
    def from_file(cls,filename):
        with open(filename,"rb") as f:
            obj = pickle.load(f)
        return obj
    
    def _gen_binning_matrix(self,X_all):
        """
        Generate a matrix which, when applied to data, produces binning statistics. Matrix has shape (X_all.shape[1],self.num_bins)

        Args:
            X_all (ndarray): coordinates to bin over.
        """
        dims = X_all.shape[1]
        num_bins = 1
        for n in range(dims):
            xn_ledge_check =  X_all[:,n,None] - self.bin_coords[n][None,:-1]
            xn_uedge_check = -X_all[:,n,None] + self.bin_coords[n][None,1:]
            xn_binning = np.sign(xn_ledge_check*xn_uedge_check) == 1
            if n > 0:
                binning = np.einsum('k...,ki->k...i',binning,xn_binning)
            else:
                binning = xn_binning.copy()
            num_bins *= xn_binning.shape[1]
        return binning.astype(int).reshape(X_all.shape[0],num_bins)
    
    @classmethod
    def with_smart_binning(cls,coord_names,variable_names,sample_dataframe,init_grid_shape,sub_divisions=1,aux_variable = None, tol=10.0):
        """
        Set up a histogram using semi-dynamically chosen bins. For now the initial grid has to be regular 
        (init_grid_shape merely specifies the initial number of subdivisions along each coordinate axis.)

        Args:
            coord_names (list): Unique labels for each dependent variable i.e. coordinate.
            variable_names (list): Unique labels for each independent variable. 
            sample_dataframe (pd.DataFrame): A dataframe, the data from which will be used to determine the binning. 
            init_grid_shape (tuple): Number of subdivisions along each coordinate axis. 
            sub_divisions (int, optional): Specify the number of subdivisions within the dynamically chosen bins. Defaults to 1.
            aux_variable (string, optional): Specify another column (variable) from the dataframe that will be used for filtering. Defaults to None.
            tol (float, optional): An aditional padding or tolerance that's added around data locations to give the final bins. Defaults to 10.0.

        Returns:
            Histogram: A histogram with semi-dynamically chosen bin_coords. 
        """
        smart_grid = optimal_grid(coord_names,sample_dataframe,init_grid_shape,sub_divisions,aux_variable,tol)
        return cls(coord_names,variable_names,*smart_grid,aux_variable=aux_variable)
    
    @staticmethod
    def _gen_mask_matrix(y_all,y_aux=None):
        mask = np.isfinite(y_all)
        if y_aux is not None:
            # keep this simple: should work for y_aux being boolean or numeric (e.g. power)
            mask *= (y_aux > 0)[:,None]
        y_all_ = y_all.copy()
        y_all_[~mask] = 0.0
        return y_all_.T, (mask).astype(int)
    
    def bin(self,X_all,y_all,y_aux=None):
        """
        Calculate binning statistics over a dataset of independent variables y_all at coordinates X_all. 

        Args:
            X_all (ndarray): Coordinates of dependent variables
            y_all (ndarray): Independent variables to bin
            y_aux (ndarray): Auxiliary criteria used to mask-out data entries. 
        """
        
        M_binning    = self._gen_binning_matrix(X_all)
        y_use,M_mask = self._gen_mask_matrix(y_all,y_aux)
        
        self.counts += np.einsum('ki,kj->ji',M_binning,M_mask)
        self.sums += y_use @ M_binning
        self.sum2 += y_use**2 @ M_binning
        
    def bin_df(self,dataframe):
        """
        Calculate binning statistics for a given dataframe. 

        Args:
            dataframe (pd.DataFrame): Dataframe containing coordinates of independent variables and corresponding dependent variables.
        """
        if self.aux_variable in dataframe.columns:
            self.bin(dataframe.loc[:,self.coord_names].values,
                    dataframe.loc[:,self.variable_names].values,
                    dataframe.loc[:,self.aux_variable].values)
        else:
            self.bin(dataframe.loc[:,self.coord_names].values,
                    dataframe.loc[:,self.variable_names].values)
        
    def __call__(self,variable_name):
        """
        Obtain a histogram with stats for just a single independent variable. 

        Args:
            variable_name (string): independent variable to return histogram for.

        Returns:
            Histogram: A histogram containing statistics for a single variable. 
        """
        sub_hist = Histogram([variable_name],self.bin_coords)
        ind = list(self.variable_names).index(variable_name)
        sub_hist.counts = np.atleast_2d(self.counts[:,ind])
        sub_hist.sums   = np.atleast_2d(self.sums[:,ind])
        sub_hist.sum2   = np.atleast_2d(self.sum2[:,ind])
        return sub_hist
    
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
        sub_hist = Histogram([coord],[],init_sub_grid)
        M = sub_hist._gen_binning_matrix(x).astype(bool)
        # Loop over all the bins on this axis, and determine the corresponding sub-grid. 
        for i,M_bin in enumerate(M.T):
            bin0_l = max(x[M_bin].min() - tol,init_sub_grid[i])
            bin0_u = min(x[M_bin].max() + tol,init_sub_grid[i+1])
            # Check for overlap with current subgrid
            if i>0:
                if bin0_l <= sub_grid[-1]:
                    sub_grid = np.r_[sub_grid,np.linspace(bin0_l,bin0_u,sub_divisions+1,endpoint=True)[1:]]
                    continue
            # Otherwise, (no overlap)
            sub_grid = np.r_[sub_grid,np.linspace(bin0_l,bin0_u,sub_divisions+1,endpoint=True)]
        grid.append(sub_grid)
    return grid
            
# %%
