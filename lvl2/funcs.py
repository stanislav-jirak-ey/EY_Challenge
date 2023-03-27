## Imports
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Visualization
import ipyleaflet
import matplotlib.pyplot as plt
from IPython.display import Image
import seaborn as sns

# Data Science
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Feature Engineering
from sklearn.model_selection import train_test_split

# Machine Learning
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score


# Planetary Computer Tools
import pystac
import pystac_client
import odc
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
from odc.stac import stac_load
import planetary_computer as pc
from odc.algo import to_rgba


# GIS
import xarray as xr
import rasterio.features
import rioxarray as rio
import xrspatial.multispectral as ms



pc.settings.set_subscription_key('bd6a9f4592344b95bff8fd62a97cb403')

# Others
import requests
import rich.table
from itertools import cycle
from tqdm import tqdm
tqdm.pandas()
from tqdm.notebook import tqdm_notebook
tqdm_notebook.pandas()
import color_maps
## Functions
#### General

def _process_data(lat, long, box_size_deg, res, res_scale):
    
    min_lon = long-box_size_deg/2
    min_lat = lat-box_size_deg/2
    max_lon = long+box_size_deg/2
    max_lat = lat+box_size_deg/2
    bounds = (min_lon, min_lat, max_lon, max_lat)
    scale = res/res_scale
    
    return bounds, scale


#### Get Sentinel-2 Data
def download_sentinel2_data(lat, long, season,verbose=False ,box_size_deg=0.1, res=20, res_scale=111320.0):
    
    '''Download's optical sentinel2 data for singular coordinate values at a specified timeframe\n
    
    Attributes:\n
        lat (float) - centroid latitude of the location\n
        long (float) = centroid longitude of the location\n
        time_slice (str) = time period in a YYYY-MM-DD/YYYY-MM-DD format\n
        box_size_deg (float) = surrounding box in degrees\n
        resolution (int) = picture resolution in meters per pixel\n
        res_scale (int) = degrees per pixel for given format, default is for CRS:4326'''
    
# 2022-01-01
# 2022-05-07

# 2022-05-01
# 2022-09-06
    if season == 'SA':
        time_slice = "2022-05-01/2022-09-06"
    if season == 'WS':
        time_slice = "2022-01-01/2022-05-07"
        
    # Process Data
    
    bounds, scale = _process_data(lat, long, box_size_deg, res, res_scale)
    
    # Search blob
    
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = stac.search(collections=["sentinel-2-l2a"], bbox=bounds, datetime=time_slice)
    items = list(search.get_all_items())
    if verbose == True:
        print('This is the number of scenes that touch our region:',len(items))
    
    # Load from blob
    
    xx = stac_load(
    items,
    bands=["red", "green", "blue", "nir","swir16", 'swir22', 'rededge', "SCL"],
    crs="EPSG:4326", # Latitude-Longitude
    resolution=scale, # Degrees
    chunks={"x": 2048, "y": 2048},
    dtype="uint16",
    patch_url=pc.sign,
    bbox=bounds
)   
    return xx
    
    
    
    
#### Get Landsat Data
def download_landsat_data(lat, long, season, box_size_deg = 0.1, res=30, res_scale=111320.0, verbose=False):
    
    '''lat_long = centroid coordinates of the location\n
        time_slice = SA/WS\n
        box_size_deg = surrounding box in degrees\n
        resolution = meters per pixel\n
        res_scale = degrees per pixel for given format, default is for CRS:4326'''
        
    if season == 'SA':
        time_slice = "2022-05-01/2022-08-31"
    if season == 'WS':
        time_slice = "2022-01-01/2022-04-30"
    
    # Process Data 
    
    bounds, scale = _process_data(lat, long, box_size_deg, res, res_scale)
    
    # Search blob
    
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = stac.search(
        collections=["landsat-c2-l2"], 
        bbox=bounds, 
        datetime=time_slice,
        query={"platform": {"in": ["landsat-8", "landsat-9"]},},
    )
    items = list(search.get_all_items())
    no_items = len(items)
    if verbose == True:
        print('This is the number of scenes that touch our region:',no_items)
    
    # Get From blob
    
    xx = stac_load(
    items,
    bands=['coastal', "blue", "green", "red", "nir08", 'swir16', 'swir22', "qa_pixel"],
    crs="EPSG:4326", # Latitude-Longitude
    resolution=scale, # Degrees
    chunks={"x": 2048, "y": 2048},
    patch_url=pc.sign,
    bbox=bounds
)
    return xx, no_items


### Sentinel Mask
def _colorize(xx, colormap):
    return xr.DataArray(colormap[xx.data], coords=xx.coords, dims=(*xx.dims, "band"))

def sentinel_mask(data,colormap=color_maps.sentinel_colormap):
    '''
    data - sentinel_data\n

    colormap = colormap object from color_maps py notebook
    '''

    cloud_mask = \
        (data.SCL != 0) & \
        (data.SCL != 1) & \
        (data.SCL != 3) & \
        (data.SCL != 6) & \
        (data.SCL != 8) & \
        (data.SCL != 9) & \
        (data.SCL != 10)
        
    # Apply cloud mask ... NO Clouds, NO Cloud Shadows and NO Water pixels
    # All masked pixels are converted to "No Data" and stored as 16-bit integers
    cleaned_data = data.where(cloud_mask).astype("uint16")

    # Load SCL band, then convert to RGB using color scheme above
    # scl_rgba_clean = colorize(cleaned_data.SCL.compute(), colormap)

    return cleaned_data


### Landsat Mask
def landsat_mask(data, flags_list=['fill', 'dilated_cloud', 'cirrus', 'cloud', 'shadow', 'water'], bit_flags = color_maps.landsat_bit_flags):
    
    # Create the result mask filled with zeros and the same shape as the mask
    mask = data['qa_pixel']
    final_mask = np.zeros_like(mask)
    
    # Loop through the flags  
    for flag in flags_list:
        
        # get the mask for each flag
        flag_mask = np.bitwise_and(mask, bit_flags[flag])
        
        # add it to the final flag
        final_mask = final_mask | flag_mask
    
    return final_mask > 0




### Landsat corruption fix

def fix_corrupted(landsat_img):
    '''Insert landsat function output
    Returns back a landsat_img without the corrupted time period'''

    errors = []
    print(len(landsat_img.time))
    for i in np.arange(len(landsat_img.time)):
    
        try :
            ### Better way to detect corrupted?
            landsat_img.isel(time=i).mean(dim=['longitude','latitude']).compute()
            
        except:
            
            err = landsat_img.isel(time=i).time.values
            errors.append(err)
    
    
    for err in errors:
        print(err)
        landsat_img = landsat_img.drop([err], dim='time')
    
    
    return landsat_img
            
def _pixel(count):
    return (1/111111)*(count/2)


import ast

## DEFINE FUNCTIONS
def convert_coordinates(df, column_name):
    """
    Converts a column of string coordinates to tuples of floats.

    Parameters:
    df (pandas.DataFrame): The dataframe to modify.
    column_name (str): The name of the column containing the string coordinates.

    Returns:
    pandas.DataFrame: The modified dataframe.
    """
    # Convert the "coordinates" column from string to tuple of floats
    def parse_coordinate_str(coord_str):
        try:
            return ast.literal_eval(coord_str)
        except SyntaxError:
            # Try parsing string without trailing parenthesis
            coord_str += ')'
            try:
                return ast.literal_eval(coord_str)
            except ValueError:
                return None

    df[column_name] = df[column_name].apply(parse_coordinate_str)
    return df

import pystac_client

def get_sentinel_data(latlong, time_slice, assets, pixel_size=3, verbose=False):
    '''
    This is a revised version of the original function!
    It takes in a function for generating bounding box around a given location.
    
    Attributes:
    latlong (tuple) - A tuple with 2 elements - latitude and longitude
    time_slice (str) - Timeframe for which the VV and VH values have to be extracted
    assets (list) - A list of bands to be extracted
    pixel_size (int) - Size of the bounding box in pixels (e.g., a pixel_size of 3 returns a 3x3 box)
    verbose (bool) - How explicit the function will be in the process 

    Returns:
    (VV, VH) - Tuple of polarization waves
    
    TODO: Use RasterIO to download 'hh', 'hv' bands as well
        RasterioIOError: '/vsicurl/https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/GRD/2020/3/21/IW/DV/S1A_IW_GRDH_1SDV_20200321T111127_20200321T111152_031773_03AA42_B7F7/measurement/iw-vh.rtc.tiff' does not exist in the file system, and is not recognized as a supported dataset name.
    '''
    # Builds a bounding box
    bbox_of_interest = _build_bounding_box(latlong, pixel_size=pixel_size)
    
    # Print additional Information
    if verbose:
        print(f'Downloading data for location {bbox_of_interest} during flyover at {time_slice}')

    # Open and search planetary computer with the specified data
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )
    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox_of_interest,
        datetime=time_slice,
        query={"platform": {"in": ["SENTINEL-1A"]},}
    )
    items = list(search.get_all_items())

        # Process for handling multiple or zero images in the selected time period
    if len(items) == 0:
        raise ValueError(f"No search results found")
    elif len(items) > 1:
        if verbose == True:
            print(f"Multiple search results found. Using first result.")
            
    scale = 10 / 111320.0
    # Load the band data and return
    #data = stac_load(items, patch_url=pc.sign, bbox=bbox_of_interest, assets=assets)
    data = stac_load(items,bands=assets, patch_url=pc.sign, bbox=bbox_of_interest, crs="EPSG:4326", resolution=scale).isel(time=0)
    data = data.mean()
    # band_data = {band: data[band].astype("float").values.tolist()[0][0] for band in assets}
    return data

from datetime import datetime, timedelta

def _generate_time_slices(start_date, end_date, freq=12):
    '''
    Generate time slices on a biweekly basis starting from the specified start_date.
    The last time slice may end after the specified end_date.
    
    Attributes:
    start_date (str): A string specifying the start date in the format "YYYY-MM-DD".
    end_date (str): A string specifying the end date in the format "YYYY-MM-DD".
    frequency (int): A string specifying the frequency of the time slices in weeks.
    
    Returns:
    list of timeslices (list)
    '''
    time_slices = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    delta = timedelta(int(freq))
    
    while current_date <= end_date:
        time_slice = f"{current_date.strftime('%Y-%m-%d')}/{(current_date+delta).strftime('%Y-%m-%d')}"
        time_slices.append(time_slice)
        current_date += delta
        
    return time_slices

def download_sentinel1_data(coordinates, season, bands=['vh', 'vv'], freq='12D', pixel_size=3, verbose=False, save=True):
    """
    Retrieves Sentinel-1 SAR data for a given crop's location and date range, and returns an XArray Dataset
    of the VH and VV polarizations.

    Parameters:
    coordinates (list): A list of tuples containing the latitude and longitude coordinates for each crop.
    start_date (str): The starting date of the date range in ISO format (yyyy-mm-dd).
    end_date (str): The ending date of the date range in ISO format (yyyy-mm-dd).
    assets (list): A list of the asset names to retrieve data for, 'vh' and 'vv'. Default is ['vh', 'vv'].
    freq (str): The frequency with which to generate time slices. Default is '12D' for biweekly slices.
    pixel_size (int) - Size of the bounding box in pixels (e.g., a pixel_size of 3 returns a 3x3 box)
    verbose (bool) - if true prints information on the data being downloaded
    netcdf (bool) - if True save the xArray in .nc format to the current working directory.
    
    Returns:
    xarray.Dataset: A Dataset containing the VH and VV polarization values for each crop's location and date slice.
    """
    
    freq = _parse_frequency(freq)
    
    sa_time_slice = "2022-05-01/2022-08-24"
    ws_time_slice = "2022-01-08/2022-04-30"
    
    start_date_sa, end_date_sa = sa_time_slice.split('/')
    start_date_ws, end_date_ws = ws_time_slice.split('/')
    date_slices_sa = _generate_time_slices(start_date_sa, end_date_sa, freq=freq)
    date_slices_ws = _generate_time_slices(start_date_ws, end_date_ws, freq=freq)
        
    
    iters = 0 # checkpoint counter
    
    for coordinate in tqdm(coordinates):
        
        
        if season[iters] == 'SA':
            date_slices = date_slices_sa
        if season[iters] == 'WS':
            date_slices = date_slices_ws
        
        
        iters += 1
        for date_slice in date_slices:
            
            try:
                if date_slice == date_slices[0]: # Create New DataFrame on the first iteration
                    
                    data = get_sentinel_data(coordinate, date_slice, bands, pixel_size = pixel_size, verbose=verbose)
                    
                else: # Append to Existing DataFrame on subsequent iterations
                
                    data_ds = get_sentinel_data(coordinate, date_slice, bands, pixel_size = pixel_size, verbose=verbose)
                    data = xr.concat([data, data_ds], dim='time')
                    
                    
                #data.append([coordinate, date_slice, data_ds['vh'], data_ds['vv']])
            except RasterioIOError as e:  # rasterio.errors.RasterioIOError
                print(f'Skipping coordinate {coordinate} for {date_slice}: {str(e)}')
                continue
            
            
            
            
        if coordinate == coordinates[0]: # Create new full_data DataFrame on the first iteration
            full_data = data.copy()
            
        else: # Append to Existing DataFrame on subsequent iterations
            full_data = xr.concat([full_data, data], dim='spatial_ref')
            
            if iters % 10 == 0: # Checkpoint on each 50th iteration
                full_data.to_netcdf(f'data_set_vh_vv_checkpoint_val.nc') # Save xArray to cwd
                print('=======================================================')
                print('Checkpoint Dataset saved!')
                

    # data_df = pd.DataFrame(data, columns=['coordinate', 'time_slice', 'vh', 'vv'])
    
    if save:
        full_data.to_netcdf(f'data_vh_vv_full_val.nc') # Save xArray to cwd
    print('=======================================================')
    print('XArray Dataset saved!')

    return full_data


import utm

def _build_bounding_box(latlong, pixel_size=3):
    '''
    Returns a bounding box of the specified size around a given latitude and longitude location
    Attributes:
    latlong - A tuple with 2 elements - latitude and longitude
    pixel_size - Size of the bounding box in pixels (e.g., a pixel_size of 3 returns a 3x3 box)
    '''
    try:
        latitude, longitude = latlong
    except (ValueError, TypeError, SyntaxError):
        raise ValueError('latlong must be a tuple with 2 elements - latitude and longitude')

    # Convert latitude and longitude to UTM coordinates
    easting, northing, zone_number, zone_letter = utm.from_latlon(latitude, longitude)

    # Calculate bounding box size in meters based on pixel size
    pixel_size_m = pixel_size * 10  # 10 meters per pixel
    half_size_m = pixel_size_m / 2

    # Calculate bounding box bounds in UTM coordinates
    bbox = (easting - half_size_m, northing - half_size_m, easting + half_size_m, northing + half_size_m)

    # Convert bounding box bounds back to latitude and longitude
    min_latitude, min_longitude = utm.to_latlon(bbox[0], bbox[1], zone_number, zone_letter)
    max_latitude, max_longitude = utm.to_latlon(bbox[2], bbox[3], zone_number, zone_letter)

    return (min_longitude, min_latitude, max_longitude, max_latitude)


        
def plot_algorithms(X_train, y_train, xlim=(0.5, 0.7), n_splits=10):
    
    '''
    Takes provided data and fits various tree-based models on it using a K-Fold cross validation, plots mean R2 scores of the various models.
    
    Parameters:
    X_train (pandas.DataFrame) - Nondependent X values used to predict the dependent value
    y_train (pandas.DataFrame) - Dependent y value to be predicted
    xlim (tuple) - sets the boundaries on the x axis for the visualization
    n_splits (int) - Number of splits for K-Fold
    
    Returns:
    None
    '''


    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, VotingRegressor
    from catboost import CatBoostRegressor
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    kfold=KFold(n_splits=n_splits)

    random_state = 42
    classifiers = []
    classifiers.append([AdaBoostRegressor(random_state=random_state),
                    RandomForestRegressor(random_state=random_state),
                    ExtraTreesRegressor(random_state=random_state),
                    XGBRegressor(random_state=random_state),
                    CatBoostRegressor(random_state = random_state),])
                    # LGBMRegressor(random_state=random_state)])


    cv_results = []
    for classifier in classifiers[0] :
        cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = 'r2', cv = kfold, n_jobs=-1))
        print(cv_results)

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame({'CrossValMeans':cv_means,'CrossValerrors': cv_std,'Algorithm':['AdaBoost',
    'RandomForest','ExtraTrees','XGB', 'CatBoost']})
    print(cv_means)
    ax = sns.barplot(data = cv_res,x='CrossValMeans',y='Algorithm', palette='colorblind',orient = 'h',**{'xerr':cv_std})
    ax.set_xlabel('Mean R2')
    ax.set_ylabel('')
    ax.set_title('CV Scores')
    plt.xlim(xlim)
        

def plot_features(features, y):
    
    
    '''
    Plots R2 score of ExtraTrees models from list of bands and indices
    
    Parameters:
    features (list of pandas.DataFrames) - list of DataFrames with different X features
    y (pandas.Series) - Values y to be predicted by the models
        
    Returns:
    None
    '''
    from sklearn.model_selection import cross_val_score, KFold, train_test_split
    from sklearn.ensemble import ExtraTreesRegressor
    
    kfold = KFold()
    trees = ExtraTreesRegressor(random_state=42)
    band_names = [df.columns[0] for df  in features]
    cv_results = []
    
    ### plot feature bars
    preds = pd.DataFrame()
    feat_importances = []
    
    for band_df in features:
        band_df = band_df.replace(0, np.nan)
        band_df = band_df.dropna(axis=1, how= 'all')
        band_df = band_df.fillna(method='ffill')
        band_df = band_df.fillna(method='bfill')
        
        
    
        X_train, X_test, y_train, y_test = train_test_split(band_df, y, test_size=0.33, random_state=42)
        trees.fit(X_train, y_train)
        band_preds = pd.Series(trees.predict(X_test))
        preds = pd.concat([preds,band_preds], axis=1)
        print(preds)
        
        cv_results.append(cross_val_score(trees, X=band_df, y=y, scoring = 'r2', cv = kfold, n_jobs=-1))
        feat_importances.append(trees.feature_importances_)
        
        
        
    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame({'CrossValMeans':cv_means,'CrossValerrors': cv_std,'Feature':band_names})
    ax = sns.barplot(data = cv_res,x='CrossValMeans',y='Feature', palette='colorblind',orient = 'h',**{'xerr':cv_std})
    ax.set_xlabel('Mean R2')
    ax.set_ylabel('')
    ax.set_title('CV Scores')
    plt.xlim((0,0.7))
    plt.show()
    
    ### plot heatmap
    
        
    preds.columns = band_names
    sns.heatmap(preds.corr(), annot=True)
    plt.show()
    
    return feat_importances
        
def get_band_stats(band, band_name=str):
    '''
    Returns original DataFrame with basic calculated statistical columns and without  columns where all values are NaN - band means from the 3 main parts of the growing cycle min, max, var, std
    
    Parameters:
    band (pandas.DataFrame) - dataframe with band values split with periods as columns
    band_name (str) - band or indice name as title for calculated columns, example: 'ndvi'
    
    Returns:
    pandas.DataFrame - 'Original DataFrame with band means from the 3 main parts of the growing cycle and min, max, var, std' for the whole dataset. Also get's rid of columns with all NaN values
    '''
    band = (band.replace(0, np.nan)
                .dropna(axis=1,how='all')
            )
    band[f'{band_name}_vegetative'] = band[band.columns[:12]].mean(axis=1)
    band[f'{band_name}_reproductive'] = band[band.columns[12:17]].mean(axis=1)
    band[f'{band_name}_ripening'] = band[band.columns[17:22]].mean(axis=1)
    band[f'{band_name}_max'] = band.max(axis=1)
    band[f'{band_name}_min'] = band.min(axis=1)
    band[f'{band_name}_var'] = band.var(axis=1)
    band[f'{band_name}_std'] = band.std(axis=1)
    band = band.reset_index(drop=True)
    
    return band
        

def _parse_frequency(freq):
    
    """
    Converts time string to number of days

    Parameters:
    time_str (str): Time period string, e.g. '12D', '2W', '1M'

    Returns:
    int: Total number of days in the given time period
    """

    time_dict = {'D': 1, 'W': 7, 'M': 30, 'Y': 365}  # conversion values for each time unit

    try:
        num = int(freq[:-1])  # extract the numeric value from the string
        unit = freq[-1]  # extract the time unit from the string

        if unit not in time_dict.keys():
            raise ValueError("Invalid time unit. Must be one of 'D', 'W', 'M', or 'Y'.")

        days = num * time_dict[unit]  # calculate the total number of days
        return days

    except ValueError as e:
        print(f"Invalid input: {format(e)}")

def info():
    print('''download_sentinel1_data' - (Radar) Downloads full sentinel1 vh/vv data from planetary computer\n
        'download_sentinel2_data' - (Optical) Download full sentinel2 images from planetary computer\n
        'download_landsat_data' - (Optical) Download full landsat images from planetary computer\n
        'sentinel_mask' - Return masked clouds, water and other atmospheric factors for sentinel imagery\n
        'landsat_mask' - Return masked clouds, water and other atmospheric factors for landsat imagery\n
        'convert_coordinates' - Converts a column of string coordinates to tuples of floats.\n
        'plot_algorithms' - Plots chart of cross validation results of several tree-based algorithms.\n
        'plot_features' - Takes a list of DataFrames with X predictors, 
        print a plot of results with the specific DataFrames and a heatmap showing correlation between y predictions of the different predictors.
        ''')
    