import numpy as np

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
        return cls.__init__(seg_x,seg_y)
        
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
                if not np.isnan(x) and not np.isnan(y):
                    if x > x_start and x < x_end and y > y_start and y < y_end:
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
                # Add new point
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
    
    