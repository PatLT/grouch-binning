import numpy as np
from matplotlib.colors import LogNorm,Normalize,SymLogNorm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

def calc_plot_range(*data_vals,non_zero=False):
    """Helper function to find a suitable plot range for plotting binned data. 

    Args:
        non_zero (bool, optional): Whether to find the minimum non-zero value. Defaults to False.

    Returns:
        float,float: Lower and upper limits of the plotting range. 
    """
    min_ = np.inf
    max_ = -np.inf
    for values in data_vals:
        values = np.asarray(values)
        new_min = np.ma.masked_equal(values,0.0,copy=True).min() if non_zero else values.min()
        new_max = values.max()
        max_ = new_max if new_max > max_ else max_
        min_ = new_min if new_min < min_ else min_
    oom = np.floor(np.log10(max_ - min_))
    return 10**oom * np.floor(min_/10**oom), 10**oom * np.ceil(max_/10**oom)

def calc_colour_range(*data_vals,non_zero=True):
    """Another helper function, this finds the upper and lower bound to the non-zero binned data. 

    Args:
        non_zero (bool, optional): Whether to find the minimum non-zero value. Defaults to True.

    Returns:
        float,float: non-zero minimum and maximum of the bin values. 
    """
    min_ = np.inf
    max_ = -np.inf
    for values in data_vals:
        new_min = np.ma.masked_equal(values,0.0,copy=True).min() if non_zero else values.min()
        new_max = values.max()
        max_ = new_max if new_max > max_ else max_
        min_ = new_min if new_min < min_ else min_
    return float(min_),float(max_)

def plot_bins(fig,variable,xy_plane_bins=None,orientation_bins=None,layer_height_bins=None,
              quantity="mean",cmap="afmhot_r",
              vrange=None,vrange_xy=None,vrange_o=None,vrange_l=None,
              colorbar=True,normalisation='linear'):
    """This function plots supplied bins (for a selection of the x-y, orientation, and Z-layer bins).

    Args:
        fig (matplotlib.figure.Figure): A matplotlib figure to append plots to. 
        variable (string): Which binned variable to plot, e.g. "Photodiode","Spatter total area".
        xy_plane_bins (DataBins, optional): Bins in the xy-plane. Defaults to None.
        orientation_bins (DataBins, optional): Bins against scan path orientation. Defaults to None.
        layer_height_bins (DataBins, optional): Bins against height / layer number. Defaults to None.
        quantity (str or callable, optional): The quantity to calculate on a per-bin basis. Defaults to "mean".
        cmap (str, optional): a matplotlib colormap. Defaults to "afmhot_r".
        vrange (tuple, optional): The range of data values to use for colours. Defaults to None.
        vrange_o (tuple, optional): The range of data values to use for orientation plot scale. Defaults to None.
        vrange_xy (tuple, optional): The range of data values to use for xy plot scale. Defaults to None.
        vrange_l (tuple, optional): The range of data values to use for layer plot scale. Defaults to None.
        colorbar (bool, optional): Whether to add a colorbar to the bottom of the plot.
        normalisation (string, linear): A normalisation option for the the colormap.

    Raises:
        ValueError: quantity arg should be 'mean', 'std', 'count', 'per_traversed', or a callable.

    Returns:
        mpl.Axes,mpl.Axes,mpl.Axes,tuple: Axex object corresponding to each subplot, and a tuple of floats corresponding to the data range. 
    """
    
    ax_widths = (3,6,6)
    ax_depth = 6
    num_cols = sum([0 if hist is None else w for hist,w in zip((xy_plane_bins,orientation_bins,layer_height_bins),ax_widths)])
    num_rows = ax_depth + 1 * colorbar
    col_ind = 1
    
    # Colormapping value ranges
    if vrange is None:
        vmin,vmax = calc_colour_range(*[hist.calculate_per_bin(quantity,variable) for hist in (layer_height_bins,xy_plane_bins,orientation_bins) 
                                        if hist is not None],
                                      non_zero=True)
    else:
        vmin,vmax = vrange
    # Select normalisation
    if normalisation.lower()=='linear':
        norm = Normalize(vmin,vmax,clip=True)
    elif normalisation.lower()=='symlog':
        norm = SymLogNorm(linthresh=np.floor(np.log10(np.abs(vmin))),vmin=vmin,vmax=vmax,clip=True)
    elif normalisation.lower()=='log':
        norm = LogNorm(vmin=vmin,vmax=vmax,clip=True)
    else:
        raise ValueError("Normalisation should be 'linear', 'symlog', or 'log'.")
    
    # Z-data_bins
    if layer_height_bins is not None:
        ax_z = fig.add_subplot(num_rows,num_cols,(col_ind,num_cols*(ax_depth-1)+ax_widths[0]))
        vrange_l = plot_layer_bins(ax_z,layer_height_bins,variable,quantity,cmap,vrange,vrange_l,norm)
        
        col_ind += ax_widths[0]
    else:
        ax_z = None
    
    # XY planar heatmap
    if xy_plane_bins is not None:
        ax_xy = fig.add_subplot(num_rows,num_cols,(col_ind,num_cols*(ax_depth-1)+col_ind+ax_widths[1]-1))
        plot_planar_bins(ax_xy,xy_plane_bins,variable,quantity,cmap,vrange,norm)
        
        col_ind += ax_widths[1]
    else:
        ax_xy = None
    
    # Orientation data_bins
    if orientation_bins is not None:
        
        ax_th = fig.add_subplot(num_rows,num_cols,(col_ind,num_cols*(ax_depth-1)+col_ind+ax_widths[2]-1),polar=True)
        vrange_o = plot_orientation_bins(ax_th,orientation_bins,variable,quantity,cmap,vrange,vrange_o,norm)
        
        col_ind += ax_widths[2]
    else:
        ax_th = None
        
    # Colorbar
    if colorbar:
        ax_cb = fig.add_subplot(num_rows,num_cols,(num_cols*ax_depth+1,num_cols*num_rows))
        fig.colorbar(ScalarMappable(cmap=cmap,norm=norm),
                    cax=ax_cb,
                    fraction=0.05,
                    aspect=40,
                    location=None,
                    orientation='horizontal')
        if normalisation.lower() == 'linear':
            ax_cb.ticklabel_format(scilimits=[-3,4])
        #col_ind += 1
    
    return ax_z,ax_xy,ax_th,(vmin,vmax),vrange_xy,vrange_o,vrange_l

def plot_orientation_bins(ax,orientation_bins,variable,
                          quantity="mean",cmap="afmhot_r",
                          vrange=None,vrange_o=None,
                          normalisation='linear'):
    """This functions plots orientation bins onto a matplotlib axis object. 
    Both must be supplied to the function. 

    Args:
        ax (matplotlib.Axes): axes to plot onto. 
        variable (string): Bin variable to access and calculate the quantity over. 
        orientation_bins (DataBins): Data bins to plot. 
        quantity (str, optional): Quantity to calculate and diplay per bin (for the selected variable). Defaults to "mean".
        cmap (str, optional): Colormap to use. Defaults to "afmhot_r".
        vrange (tuple, optional): Value range for the colormap. Defaults to None.
        vrange_o (tuple, optional): Value range for the axis. Defaults to None.
        normalisation (str, optional): Normalisation to apply to the colormap. Defaults to 'linear'.

    Raises:
        ValueError: If the supplied norm is not one of linear, symlog, or log. 

    Returns:
        tuple: Minimum and maximum values used in plot axis. 
    """
    # Colormapping
    cmap = plt.colormaps.get(cmap)
    cmap.set_under("w")
    if vrange is None:
        vrange = calc_colour_range(orientation_bins.calculate_per_bin(quantity,variable),non_zero=True)
    vmin,vmax = vrange
    # Select normalisation
    if callable(normalisation):
        norm = normalisation 
    elif normalisation.lower()=='linear':
        norm = Normalize(vmin,vmax,clip=False)
    elif normalisation.lower()=='symlog':
        norm = SymLogNorm(linthresh=np.floor(np.log10(np.abs(vmin))),vmin=vmin,vmax=vmax,clip=False)
    elif normalisation.lower()=='log':
        norm = LogNorm(vmin=vmin,vmax=vmax,clip=False)
    else:
        raise ValueError("Normalisation should be 'linear', 'symlog', or 'log'.")
    
    # Some plotting parameters
    bot = 30.0
    top = 100.0
    
    values = orientation_bins.calculate_per_bin(quantity,variable)[1:-1]
    if vrange_o is None:
      min_,max_ = calc_plot_range(values)
    else:
        min_,max_ = vrange_o
    # Make plot 
    ax.bar(0.5*(orientation_bins.bin_edges[0][1:] + orientation_bins.bin_edges[0][:-1]),
        bot + (top-bot) * (values - min_)/(max_ - min_),
        width = orientation_bins.bin_edges[0][1:] - orientation_bins.bin_edges[0][:-1],
        bottom = 0.0,
        color=cmap(norm(values))
    )
    ax.set_rticks([bot,0.5*(bot+top),top],labels=map("{:.2e}".format,[min_,0.5*(min_+max_),max_]))
    ax.set_rlabel_position(35.0)
    
    return min_,max_
    
def plot_planar_bins(ax,xy_plane_bins,variable,
                     quantity="mean",cmap="afmhot_r",
                     vrange=None,normalisation='linear'):
    """This functions plots planar (typically XY plane) bins onto a matplotlib axis object. 
    Both must be supplied to the function. 

    Args:
        ax (matplotlib.Axes): axes to plot onto. 
        variable (string): Bin variable to access and calculate the quantity over. 
        orientation_bins (DataBins): Data bins to plot. 
        quantity (str, optional): Quantity to calculate and diplay per bin (for the selected variable). Defaults to "mean".
        cmap (str, optional): Colormap to use. Defaults to "afmhot_r".
        vrange (tuple, optional): Value range for the colormap. Defaults to None.
        normalisation (str, optional): Normalisation to apply to the colormap. Defaults to 'linear'.

    Raises:
        ValueError: If the supplied norm is not one of linear, symlog, or log. 

    Returns:
        tuple: Minimum and maximum values used. 
    """
    # Colormapping
    cmap = plt.colormaps.get(cmap)
    cmap.set_under("w")
    if vrange is None:
        vrange = calc_colour_range(xy_plane_bins.calculate_per_bin(quantity,variable),non_zero=True)
    vmin,vmax = vrange
    # Select normalisation
    if callable(normalisation):
        norm = normalisation 
    elif normalisation.lower()=='linear':
        norm = Normalize(vmin,vmax,clip=False)
    elif normalisation.lower()=='symlog':
        norm = SymLogNorm(linthresh=np.floor(np.log10(np.abs(vmin))),vmin=vmin,vmax=vmax,clip=False)
    elif normalisation.lower()=='log':
        norm = LogNorm(vmin=vmin,vmax=vmax,clip=False)
    else:
        raise ValueError("Normalisation should be 'linear', 'symlog', or 'log'.")
    
    values = xy_plane_bins.calculate_per_bin(quantity,variable)[1:-1,1:-1]
    # Make plot 
    for i,(x_l,x_u) in enumerate(zip(xy_plane_bins.bin_edges[0][:-1],(xy_plane_bins.bin_edges[0][1:]))):
        for j,(y_l,y_u) in enumerate(zip(xy_plane_bins.bin_edges[1][:-1],(xy_plane_bins.bin_edges[1][1:]))):
            v = values[i,j]
            ax.fill_between([x_l,x_u],[y_l,y_l],[y_u,y_u],color=cmap(norm(v)))
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_aspect("equal")
    ax.ticklabel_format(scilimits=[4,4])
    
    return vrange
    
def plot_layer_bins(ax,layer_height_bins,variable,
                    quantity="mean",cmap="afmhot_r",
                    vrange=None,vrange_l=None,
                    normalisation='linear'):
    """This functions plots layer / height / Z-direction bins onto a matplotlib axis object. 
    Both must be supplied to the function. 

    Args:
        ax (matplotlib.Axes): axes to plot onto. 
        variable (string): Bin variable to access and calculate the quantity over. 
        orientation_bins (DataBins): Data bins to plot. 
        quantity (str, optional): Quantity to calculate and diplay per bin (for the selected variable). Defaults to "mean".
        cmap (str, optional): Colormap to use. Defaults to "afmhot_r".
        vrange (tuple, optional): Value range for the colormap. Defaults to None.
        vrange_l (tuple, optional): Value range for the plot axis. Defaults to None.
        normalisation (str, optional): Normalisation to apply to the colormap. Defaults to 'linear'.

    Raises:
        ValueError: If the supplied norm is not one of linear, symlog, or log. 

    Returns:
        tuple: Minimum and maximum values used. 
    """
    # Colormapping
    cmap = plt.colormaps.get(cmap)
    cmap.set_under("w")
    if vrange is None:
        vrange = calc_colour_range(layer_height_bins.calculate_per_bin(quantity,variable),non_zero=True)
    vmin,vmax = vrange
    # Select normalisation
    if callable(normalisation):
        norm = normalisation 
    elif normalisation.lower()=='linear':
        norm = Normalize(vmin,vmax,clip=False)
    elif normalisation.lower()=='symlog':
        norm = SymLogNorm(linthresh=np.floor(np.log10(np.abs(vmin))),vmin=vmin,vmax=vmax,clip=False)
    elif normalisation.lower()=='log':
        norm = LogNorm(vmin=vmin,vmax=vmax,clip=False)
    else:
        raise ValueError("Normalisation should be 'linear', 'symlog', or 'log'.")
    
    values = layer_height_bins.calculate_per_bin(quantity,variable)[1:-1]
    # Make plot
    ax.barh(0.5*(layer_height_bins.bin_edges[0][1:] + layer_height_bins.bin_edges[0][:-1]),
        values,
        layer_height_bins.bin_edges[0][1:] - layer_height_bins.bin_edges[0][:-1],
        color=cmap(norm(values))
    )
    ax.set_ylabel("layer number")
    if vrange_l is None:
        min_,max_ = calc_plot_range(values)
    else:
        min_,max_ = vrange_l
    ax.set_xlim(min_,max_)
    ax.set_ylim(layer_height_bins.bin_edges[0][0],layer_height_bins.bin_edges[0][-1])
    ax.tick_params(axis='x',labelrotation=-45)
    ax.ticklabel_format(axis='x',scilimits=[-3,4])
    
    return min_,max_