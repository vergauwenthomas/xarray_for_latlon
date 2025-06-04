import logging
import warnings

from xarray import DataArray, Dataset
import numpy as np

from sklearn.neighbors import BallTree


def sel_nearest_latlon(xrobj: Dataset | DataArray,
                       indexers_equivalent: dict, #NOt in index
                       metric: str = 'haversine',
                       balltree_leaf_size=40,
                       spatial_x_dim: str = 'x',
                       spatial_y_dim: str = 'y',
                       spatial_lat_name: str = 'lat', #Needed because haversine needs lat first pairs!
                       spatial_lon_name: str = 'lon', #Needed because haversine needs lat first pairs!
                       **selkwargs):
    
    #Test indexers compatibility
    

    for coordname in indexers_equivalent.keys():
        if coordname not in [spatial_lat_name, spatial_lon_name]:
            raise ValueError(f'{coordname} is not a in spatial_lat_name, or spatial_lon_name: [{spatial_lat_name}, {spatial_lon_name}]')
        if coordname in xrobj.indexes.keys():
            raise ValueError(f'{coordname} cannot be used, since it is an index level. Use .sel() for this dimension.')
    
    #test if spatial dimensions are in the index
    if spatial_x_dim not in xrobj.indexes.keys():
        raise ValueError(f'{spatial_x_dim} is not an index coordinate. Set it as an index or update the spatial_x_dim argument.')
    if spatial_y_dim not in xrobj.indexes.keys():
        raise ValueError(f'{spatial_y_dim} is not an index coordinate. Set it as an index or update the spatial_y_dim argument.')
    
    #test if lat and lon coordinates are present
    if spatial_lat_name not in xrobj.coords.keys():
        raise ValueError(f'{spatial_lat_name} is not a coordinate. Update the spatial_lat_name argument.')
    if spatial_lon_name not in xrobj.coords.keys():
        raise ValueError(f'{spatial_lon_name} is not a coordinate. Update the spatial_lon_name argument.')


    #Construct a BallTree indexer
    latdata = xrobj[spatial_lat_name].data
    londata = xrobj[spatial_lon_name].data

    #test if latlon is in degrees
    # if not test_degree_units()
    #Note: the order of lat and lon is important !! 
    d3comb = np.stack((latdata, londata), axis=-1) #as a 3D array= [[lats-2D], [lons-2D]]
    d1comb = d3comb.reshape(-1, d3comb.shape[-1]) #as a 1D array of pairs

    if metric=='haversine':
        if latdata.max() <= 2*np.pi:
            warnings.warn(f'The max latitude: {latdata.max()} <= 2*pi, they are assumed to be in DEGREES.')
        if londata.max() <= 2*np.pi:
            warnings.warn(f'The max longitude: {londata.max()} <= 2*pi, they are assumed to be in DEGREES.')

        #Create an indexer, with 'haversine' metric
        index = BallTree(np.deg2rad(d1comb), 
                    leaf_size=balltree_leaf_size, 
                    metric='haversine', #haversine approx metric for latlon space
                    #other arguments are passed to the metric
                    )

    else:
        index = BallTree(d1comb, 
                    leaf_size=balltree_leaf_size, #default
                    metric=metric, 
                    )

    
    trglat = indexers_equivalent[spatial_lat_name]
    trglon = indexers_equivalent[spatial_lon_name]
    
    #set target (order must be equal the order for the baltree input)
    trgpoint = [[trglat,
                trglon]] 

    if metric=='haversine':
        if trglat <= 2*np.pi:
            warnings.warn(f'The target latitude: {trglat} <= 2*pi, they are assumed to be in DEGREES.')
        if trglon <= 2*np.pi:
            warnings.warn(f'The target longitude: {trglon} <= 2*pi, they are assumed to be in DEGREES.')
        #to radians
        trgpoint= np.deg2rad(trgpoint)

    #find nearset index to target locations
    dist, trgidx1D = index.query(
                            trgpoint, 
                            k=1, #only the Neirest-neighbour
                            return_distance=True)
    #The index for the 1D representation of the grid
    trgidx1D = trgidx1D.ravel()[0] #unravel to integer

    #Use the index of the 1D rep to get the corresponding  X and Y values.
    
    # d3comb = np.stack((latdata, londata), axis=-1)
    i,j = np.unravel_index(trgidx1D, d3comb.shape[:-1]) #i an j are in numpy array space
    x_index = xrobj[spatial_x_dim].data[i]
    y_index = xrobj[spatial_y_dim].data[j]


    return xrobj.sel({spatial_x_dim: x_index,
                    spatial_y_dim:y_index},
                    **selkwargs)

