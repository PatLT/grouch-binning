import numpy as np
from scipy.special import binom

class Shape():
    def __init__(self,segments_x,segments_y) -> None:
        self.seg_x, self.seg_y = segments_x, segments_y
        self.com_x,self.com_y = self.com(self.seg_x,self.seg_y)
        self.num_edges = (~np.isnan(segments_x)).sum()//2
        
    @classmethod
    def from_cyclic_vertices(cls,*vertices):
        """constructor for a set of vertices defining the shape in a cyclic fashion. 

        Returns:
            shape: The shape described by these vertices.
        """
        seg_x,seg_y = cls.seg_rep_from_verts(*vertices)
        return cls(seg_x,seg_y)
        
    def inner_shape(self,dist):
        """Generate a new shape which corresponds to the older shape with its edges
        moved distance dist inside. 

        Args:
            dist (float): distance to move edges by.

        Returns:
            Shape: Inner shape. 
        """
        new_edges = []
        for edge_x_rep,edge_y_rep in zip(self.seg_x,self.seg_y):
            edge_x = edge_x_rep[~np.isnan(edge_x_rep)]
            edge_y = edge_y_rep[~np.isnan(edge_y_rep)]
            a,b,c = self.seg2edge(*edge_x,*edge_y)
            if np.abs(a*self.com_x+b*self.com_y-c-dist*np.sqrt(a**2+b**2)) < np.abs(a*self.com_x+b*self.com_y-c+dist*np.sqrt(a**2+b**2)):
                c_new = c + dist*np.sqrt(a**2+b**2)
            else:
                c_new = c - dist*np.sqrt(a**2+b**2)
            new_edges.append((a,b,c_new))
        # Segment representation
        new_seg_x,new_seg_y = self.seg_rep_from_edges(*new_edges)
        # Now remove redundant intersections to construct vertices. 
        # However, we cannot assume that these are the same as the original shape.
        trial_seg_x = new_seg_x.copy() ; trial_seg_x[np.isnan(self.seg_x)] = np.nan
        trial_seg_y = new_seg_y.copy() ; trial_seg_y[np.isnan(self.seg_y)] = np.nan
        # So loop over these trial edges and check that none of the calculated 
        #... intersections are contained within said edges.
        pos_new_verts = []
        new_verts = [] # Move index pairs between these lists as they are found.
        for i,(edge_x_rep,edge_y_rep,intersec_x,intersec_y) in enumerate(zip(trial_seg_x,trial_seg_y,new_seg_x,new_seg_y)):
            edge_x = edge_x_rep[~np.isnan(edge_x_rep)]
            x_start = min(edge_x) ; x_end = max(edge_x)
            edge_y = edge_y_rep[~np.isnan(edge_y_rep)]
            y_start = min(edge_y) ; y_end = max(edge_y)
            # Check intersections
            for j,(x,y) in enumerate(zip(intersec_x,intersec_y)):
                if not np.isnan(x) and not np.isnan(y) and j!=i:
                    if x >= x_start and x <= x_end and y >= y_start and y <= y_end:
                        # Have found an intersection that lies on this line segment.
                        # If it lies on another line segment, it's a new vertex. 
                        if [j,i] in pos_new_verts:
                            new_verts.append([i,j])
                        else:
                            pos_new_verts.append([i,j])
        # Now check which vertices / edge need to be removed, and add new ones.
        if len(new_verts) > 0:
            vert_to_remove = []
            edge_to_remove = []
            for i,j in new_verts:
                # Removing a point means removing an edge.
                for ind,current_edge in enumerate(trial_seg_x):
                    loc0,loc1 = np.arange(self.num_edges)[~np.isnan(current_edge)]
                    if (not np.isnan(trial_seg_x[i,loc0]) and not np.isnan(trial_seg_x[j,loc1])) or (not np.isnan(trial_seg_x[i,loc1]) and not np.isnan(trial_seg_x[j,loc0])):
                        # Have found a vertex to remove
                        vert_to_remove.append([loc0,loc1])
                        edge_to_remove.append(ind)
            # Add new points in a separate loop
            for i,j in new_verts:
                trial_seg_x[i,j] = new_seg_x[i,j]
                trial_seg_x[j,i] = new_seg_x[j,i]
                trial_seg_y[i,j] = new_seg_y[i,j]
                trial_seg_y[j,i] = new_seg_y[j,i]
            # Remove points
            for l,k in vert_to_remove:
                trial_seg_x[l,k] = np.nan ; trial_seg_x[k,l] = np.nan
                trial_seg_y[l,k] = np.nan ; trial_seg_y[k,l] = np.NaN
            trial_seg_x = np.delete(trial_seg_x,edge_to_remove,axis=0)
            trial_seg_x = np.delete(trial_seg_x,edge_to_remove,axis=1)
            trial_seg_y = np.delete(trial_seg_y,edge_to_remove,axis=0)
            trial_seg_y = np.delete(trial_seg_y,edge_to_remove,axis=1)
            
        return Shape(trial_seg_x,trial_seg_y)
    
    def get_cyclic_vert_order(self,complete_loop=True):
        verts = []
        cnt = 0 ; i = 0 ; i_prev = None
        while cnt < self.num_edges:
            if cnt%2==0:
                line_x = self.seg_x[i,:] ; line_y = self.seg_y[i,:]
            else:
                line_x = self.seg_x[:,i] ; line_y = self.seg_y[:,i]
            inds = np.argwhere(~np.isnan(line_x)).flatten()
            if i_prev is None:
                i_prev = inds[0]
            verts.append([line_x[i_prev],line_y[i_prev]])
            # New values of i, i_prev
            i_temp = i
            i = inds[np.where(inds!=i_prev)][0]
            i_prev = i_temp
            cnt += 1
        # Go back to first vertex?
        if complete_loop:
            verts.append(verts[0])
        return verts
            
        
        
    @staticmethod
    def seg_rep_from_verts(*vertices):
        """Given a set of vertices (x,y) arranged in cyclical order, construct a
        segment representation i.e. an array where each row or column represents an
        edge, and each entry represents an intersection vertex. 

        Returns:
            ndarray,ndarray: Arrays corresponding to x and y coordinates of vertices.
        """
        num_verts = len(vertices)
        edges_x = np.empty((num_verts,num_verts),dtype=float)
        edges_x.fill(np.nan)
        edges_y = edges_x.copy()
        for i,(x_i,y_i) in enumerate(vertices):
            edges_x[(i-1)%num_verts,i] = x_i
            edges_x[i,(i-1)%num_verts] = x_i
            edges_y[(i-1)%num_verts,i] = y_i
            edges_y[i,(i-1)%num_verts] = y_i
        return edges_x,edges_y
    
    @staticmethod
    def seg_rep_from_edges(*edges):
        """Construct a segment representation from edges instead of vertices. 

        Returns:
            ndarray,ndarray: Arrays corresponding to x and y coordinates of vertices.
        """
        num_edges = len(edges)
        edges_x = np.empty((num_edges,num_edges),dtype=float)
        edges_x.fill(np.nan)
        edges_y = edges_x.copy()
        for i,edge_i in enumerate(edges):
            # Per EDGE
            for j,edge_j in enumerate(edges):
                if i<j:
                    x_ij,y_ij = Shape.intersection(edge_i,edge_j)
                    if x_ij is not None and y_ij is not None:
                        edges_x[i,j] = x_ij ; edges_x[j,i] = x_ij
                        edges_y[i,j] = y_ij ; edges_y[j,i] = y_ij

        return edges_x,edges_y
        
    @staticmethod
    def intersection(edge1,edge2):
        """Intersection of two infinite lines.
        
        Args:
            edge1 (tuple,float): An infinite edge defined by 3 floats (ax+by=c).
            edge2 (tuple float): An(other) infinite edge defined by 3 floats (ax+by=c).

        Returns:
            tuple: Coordinates x,y of the intersection. 
        """
        a1,b1,c1 = edge1
        a2,b2,c2 = edge2
        denom = a1*b2-a2*b1
        if denom == 0.0:
            # Lines are parallel
            return None,None
        else:
            return (c1*b2 - c2*b1)/denom, (a1*c2 - a2*c1)/denom
        
    @staticmethod
    def seg2edge(x1,x2,y1,y2):
        """Convert a segment spanning points p1 to p2 into an infinite edge
        given by a,b,c where ax+by=c. 

        Args:
            x1 (float): x coord of 1st point
            x2 (float): x coord of 2nd point
            y1 (float): y coord of 1st point
            y2 (float): y coord of 2nd point

        Returns:
            tuple: a,b,c rep of an infinite edge
        """
        th = np.arctan2(y1-y2,x2-x1)
        a = np.sin(th) ; b = np.cos(th) ; c = a*x1 + b*y1
        return a,b,c
    
    @staticmethod
    def seg_intersect(edge,vert1,vert2):
        """Intersection of a line (edge) with a line segment between vert1, vert2. 

        Args:
            edge (tuple,float): An infinite edge defined by 3 floats (ax+by=c). 
            vert1 (tuple,float): Vertex at one end of the line segment
            vert2 (tuple,float): Vertex at other end of line segment
        """
        x1,y1 = vert1
        x2,y2 = vert2
        # Determine a,b,c for line segment
        a_,b_,c_ = Shape.seg2edge(x1,x2,y1,y2)
        x,y = Shape.intersection(edge,(a_,b_,c_))
        if x is not None and y is not None:
            if x <= max(x1,x2) and x >= min(x1,x2) and y <= max(y1,y2) and y >= min(y1,y2):
                return x,y
            
        return None,None
        
    @staticmethod
    def com(sgr_x,sgr_y):
        """Calculate the centre of mass (CoM) of a set of points, 
        provided in segment representation. 

        Args:
            sgr_x (ndarray): Segment representation of points/line for x coordinates.
            sgr_y (ndarray): Segment representation of points/line for y coordinates.

        Returns:
            tuple: x,y coordinates of centre of mass
        """
        com_x = np.ma.masked_invalid(sgr_x).sum()/(~np.isnan(sgr_x)).sum()
        com_y = np.ma.masked_invalid(sgr_y).sum()/(~np.isnan(sgr_y)).sum()
        return com_x,com_y
    
class Slice(Shape):
    def __init__(self, segments_x, segments_y) -> None:
        super().__init__(segments_x, segments_y)
        self.contours = []
        self.scan_path = []
        self.scan_cmds = []
    
    def gen_scan_path(self,theta,dist,**kwargs):
        """Generate scan paths for this slice. 

        Args:
            theta (float): Orientation of the hatch lines
            dist (float): Separation between scan lines
        """
        contour_scans = kwargs.get("contour_scans",1)
        hta = kwargs.get("hatch_turn_allowance",kwargs.get("hta",1.5))
        smooth_inner_contour = kwargs.get("smooth_inner_contour",True)
        # Inner contour
        if contour_scans > 0:
            self.contours.append(self.inner_shape(dist))
            for i in range(1,contour_scans):
                self.contours.append(self.contours[i-1].inner_shape(dist))
            innermost = self.contours[-1]
        else:
            innermost = self
        # Hatch boundary shape
        hatch_bshape = innermost.inner_shape(dist*hta)
        self._hbs = hatch_bshape
        # a,b,c defining the hatch lines
        a_h = -np.sin(theta)
        b_h = np.cos(theta)
        c_h_array = a_h*hatch_bshape.seg_x + b_h*hatch_bshape.seg_y
        inds = np.triu_indices_from(c_h_array,k=1)
        c_h_all = c_h_array[inds][~np.isnan(c_h_array[inds])]
        dists = np.abs(c_h_all[:,None] - c_h_all[None,:])
        c_h_i,c_h_f = c_h_all[np.array(np.unravel_index(np.argmax(dists),dists.shape))]
        #  Hatch sequence
        direction = 1
        hatch_pts_x = []
        hatch_pts_y = []
        # Loop over every line in the hatch 
        for c_h in np.arange(min(c_h_i,c_h_f),max(c_h_i,c_h_f),dist)[1:]:
            # Intersections of this hatch line
            inters_x = [] ; inters_y = []
            for i in range(innermost.num_edges):
                x1 , x2   = innermost.seg_x[i][~np.isnan(innermost.seg_x[i])]
                y1 , y2   = innermost.seg_y[i][~np.isnan(innermost.seg_y[i])]
                x1_i,x2_i = hatch_bshape.seg_x[i][~np.isnan(hatch_bshape.seg_x[i])]
                y1_i,y2_i = hatch_bshape.seg_y[i][~np.isnan(hatch_bshape.seg_y[i])]
                x_h , y_h   = Shape.seg_intersect((a_h,b_h,c_h),(x1,y1),(x2,y2))
                x_h_i,y_h_i = Shape.seg_intersect((a_h,b_h,c_h),(x1_i,y1_i),(x2_i,y2_i))
                if x_h is not None and y_h is not None:
                    inters_x.append(x_h)
                    inters_y.append(y_h)
                if x_h_i is not None and y_h_i is not None:
                    inters_x.append(x_h_i)
                    inters_y.append(y_h_i)
            if np.abs(theta) != np.pi/2:
                order_ = np.argsort(inters_x)
            else:
                order_ = np.argsort(inters_y)
            inters_x = np.array(inters_x)[order_][::direction]
            inters_y = np.array(inters_y)[order_][::direction]
            direction *= -1
            hatch_pts_x = np.r_[hatch_pts_x,inters_x]
            hatch_pts_y = np.r_[hatch_pts_y,inters_y]
        self.hatch_pts_x = hatch_pts_x[:-1]
        self.hatch_pts_y = hatch_pts_y[:-1]
        # The scan path is specified as a set of coordinates
        # AND a set of commands that specify how to sample the coordinates.
        # Commands:
        # c = 1+ -> Start of Bezier curve of degree c
        # c = 0  -> Laser switched off
        # c = -1 -> Control point for bezier curve.
        
        # Outer perimeter
        self.scan_path += self.get_cyclic_vert_order()
        # Contour Scans
        if contour_scans > 0:
            for contour in self.contours:
                self.scan_path += contour.get_cyclic_vert_order()
        # Hatching
        self.scan_path += list(zip(self.hatch_pts_x,self.hatch_pts_y))
        
        self.scan_path = np.array(self.scan_path)
       
        
        
def bezier(d,*control_points,**kwargs):
    """Generate a Bezier curve, sampling points separated by arclength d. 
    The degree of the Bezier curve is arbitrary, and determined by the 
    number of control points.

    Args:
        d (float): Sampling distance. 

    Returns:
        ndarray: Bezier curve sampled at even interval along the arclength.
    """
    num_t_samples = kwargs.get("num_t_samples",100)
    n = len(control_points)
    bz_func = lambda t: sum(binom(n,i)*(1-t)**(n-i)*t**i*control_points[i][None,:] for i in range(n))
    # Now take lazy approach: generate 100 sample points then resample as required...
    t_ = np.linspace(0.0,1.0,num_t_samples)
    samples = bz_func(t_[:,None])
    arc_dists = np.r_[[0.0],np.cumsum(np.linalg.norm(samples[1:] - samples[:-1],axis=1))]
    t_final = np.interp(np.arange(0.0,arc_dists[-1],d),arc_dists,t_)[:,None]
    return bz_func(t_final)