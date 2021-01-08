"""
Implementatio of tile coding (chapter 9.5.4) for example 10.1 of "Reinforcement
learning" by Sutton and Barto
"""

import numpy as np


class feature_vecs():
    """Tile coding for the Mountain car example 10.1 from Sutton and Barto's
    "Reinforcement Learning" """
    def __init__(self, dims, nr_tilings, tiles_per_dim, displ_vecs):
       self.dims = dims
       self.nr_tilings = nr_tilings
       self.tiles_per_dim = tiles_per_dim
       self.displ_vecs = displ_vecs
       
       self.nr_dims = len(self.dims)
       self.tile_width = np.array([(dim[1] - dim[0])/self.tiles_per_dim 
                                   for dim_idx, dim in enumerate(self.dims)])
       self.displ_unit = np.array([self.tile_width[dim_idx]/self.nr_tilings 
                                  for dim_idx, dim in enumerate(self.dims)])
       self.tiling_displ = [displ_vecs[tiling]*self.displ_unit 
                            for tiling in range(self.nr_tilings)] 
        
       self.idcs_per_action = self.nr_tilings*self.tiles_per_dim**self.nr_dims
       
    def calc_tile_idx(self, state,  tiling):
        """
        Determine tile index for current state and tiling

        Parameters
        ----------
        state : list
            current state of agent
        tiling : int
            current tiling

        Returns
        -------
        tile_idx : ndarray
            index of tile per dimension of state

        """
        tile_idx = np.zeros(len(self.dims), dtype=int)
        for dim_idx, dim in enumerate(self.dims):
    
            dim_discr = np.linspace(dim[0] + self.tile_width[dim_idx]/2, 
                                    dim[1] - self.tile_width[dim_idx]/2, 
                                    self.tiles_per_dim) 
            
            dim_discr += self.tiling_displ[tiling][dim_idx]
            
            dist = np.abs(state[dim_idx] - dim_discr)
            tile_idx[dim_idx] = int(np.argmin(dist))
    
        return tile_idx
    
    def calc_feature_vec(self, state, action = 0):
        """
        Determine feature vector for given state 

        Parameters
        ----------
        state : list
            current state of agent
        action : int, optional
            current action taken by agent. The default is 0.

        Returns
        -------
        fvec_idx_per_tiling : ndarray
            feature vector for current state

        """
        fvec_idx_per_tiling = np.zeros((self.nr_tilings)).astype(dtype=int)              
    
        for tiling in range(self.nr_tilings):
            tile_idx = self.calc_tile_idx(state, tiling)
            fvec_idx_per_tiling[tiling] = np.prod(tile_idx) + (tiling)*self.tiles_per_dim**self.nr_dims
            
        fvec_idx_per_tiling += action*self.idcs_per_action
            
        return fvec_idx_per_tiling


if __name__ == "__main__":
    #%% Example
    
    dim1 = np.array((-1.2, 0.5))
    dim2 = np.array((-0.07, 0.07))
    
    dims = np.array((dim1, dim2))
    
    nr_tilings = 8
    tiles_per_dim = 8
    action = 1
    
    displ_vecs = np.array([(1, 3), (3, 1), (-1,3), (3, -1),
                  (1,-3), (-3,1), (-1,-3), (-3,-1)], dtype="float")
    fvecs = feature_vecs(dims, nr_tilings, tiles_per_dim, displ_vecs)
    
    state = np.array([np.random.uniform(dims[0][0], dims[0][1]), 
                      np.random.uniform(dims[1][0], dims[1][1])])
    fvec_idx_per_tiling = fvecs.calc_feature_vec(state, action)
    
    print("state = " + str(state))
    print(fvec_idx_per_tiling)