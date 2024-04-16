# Binning analysis
The basic idea of the binning code is to visualise the distribution of different measured _variables_ (metrics) e.g. photodiode intensity, spatter, etc. by binning them against different _coordinates_ (independent variables) e.g. spatial position, scan orientation, etc. 
The resulting bins can be visualised in different ways by plotting different quantities per bin such as the mean, count, or standard deviation for a given variable. 
The easiest way to get what this is about is to look at [the notebook](binning_v2.ipynb). 

The code only has one unusual dependency which is the `numba` library. It's recommended to install this into a clean environment as it's prone to breaking pre-existing environments with its complicated dependencies. 

The key parts of the code are:
* The `DataBins` object, which stores all the data for a particular coordinate or set of coordinates in bins.
* The `process_dataframe` function, which takes a pandas dataframe and any number of `DataBins` instances as input, then carries out the appropriate binning over the dataframe.

The rest of the code in [fast_binning.py](fast_binning.py) basically consists of various helper functions.
* `optimal_grid` can be used to generate appropriate bins given an initial user 'guess'. This is helpful for binning against 'sparse' coordinates, such as XY plane coordinates (for certain builds).
* `expand_df` adds some common 'derived' variables and coordinates into an initial dataframe: orientation, speed, acceleration, distance traversed.
* `plot_bins` produces a quick plot from 3 `DataBins` with coordinates corresponding to build height, XY position, and scan path orientation. 
