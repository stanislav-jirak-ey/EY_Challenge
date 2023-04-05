## IMPORT LIBRARIES
import ast
import utm
import pystac_client
from datetime import datetime, timedelta
from rasterio import RasterioIOError
from tqdm import tqdm
tqdm.pandas()
import xarray as xr

# Planetary Computer Tools
import pystac
import pystac_client
import odc
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
from odc.stac import stac_load
import planetary_computer as pc
pc.settings.set_subscription_key('100bbb6552514f27a2e52ef1be8d93b6')  # SJ's primary key


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
    bounding_box_func (function) - A function that returns the bounding box of a location given its latitude and longitude.

    Returns:
    (VV, VH) - Tuple of polarization waves
    
    TODO: Use RasterIO to download 'hh', 'hv' bands as well
        RasterioIOError: '/vsicurl/https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/GRD/2020/3/21/IW/DV/S1A_IW_GRDH_1SDV_20200321T111127_20200321T111152_031773_03AA42_B7F7/measurement/iw-vh.rtc.tiff' does not exist in the file system, and is not recognized as a supported dataset name.
    '''
    # Builds a bounding box
    bbox_of_interest = _build_bounding_box(latlong, pixel_size)

    # Print additional Information
    if verbose:
        print(f'Downloading data for location {bbox_of_interest} during flyover at {time_slice}')

    # Open and search planetary computer with the specified data
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox_of_interest,
        datetime=time_slice,
        query={"platform": {"in": ["SENTINEL-1A"]},},
    )
    items = list(search.get_all_items())
    print(items)

    # Process for handling multiple or zero images in the selected time period
    if len(items) == 0:
        raise ValueError(f"No search results found")
    elif len(items) > 1:
        if verbose:
            print(f"Multiple search results found. Using first result.")

    # Load the band data and return
    data = stac_load([items[0]], patch_url=pc.sign, bbox=bbox_of_interest, assets=assets)
    return data


def _parse_frequency(freq) -> int:
    """
    Converts time string to number of days

    Parameters:
    freq (str): Time period string, e.g. '12D', '2W', '1M'

    Returns:
    int: Total number of days in the given time period
    """
    time_dict = {'D': 1, 'W': 7, 'M': 30, 'Y': 365}  # conversion values for each time unit

    try:
        num = int(freq[:-1])  # extract the numeric value from the string
        unit = freq[-1].upper()  # extract the time unit from the string and convert to uppercase

        if unit not in time_dict.keys():
            raise ValueError("Invalid time unit. Must be one of 'D', 'W', 'M', or 'Y'.")

        days = num * time_dict[unit]  # calculate the total number of days
        return days

    except (ValueError, TypeError) as e:
        print(f"Invalid input: {e}")
        return None


def download_sentinel1_data(coordinates, start_date, end_date, bands=['vh', 'vv'], freq='12D', pixel_size=3, verbose=False, save=True):
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
    
    date_slices = _generate_time_slices(start_date, end_date, freq=freq)
    
    # Initialize data variable with data for first coordinate and first time slice
    data = get_sentinel_data(coordinates[0], date_slices[0], bands, pixel_size=pixel_size, verbose=verbose)
    # Calculate the mean of the data across the sample region
    data = data.mean(dim=['coordinates']).compute()
    # Calculate RVI
    dop = (data.vv / (data.vv + data.vh))
    m = 1 - dop
    data['rvi'] = (np.sqrt(dop))*((4*data.vh)/(data.vv + data.vh))
                
    
    for coordinate in tqdm(coordinates[1:]):
        for date_slice in date_slices:
            try:
                ds = get_sentinel_data(coordinate, date_slice, bands, pixel_size=pixel_size, verbose=verbose)
                # Calculate the mean of the data across the sample region
                ds = ds.mean(dim=['coordinates']).compute()
                # Calculate RVI
                dop = (ds.vv / (ds.vv + ds.vh))
                m = 1 - dop
                ds['rvi'] = (np.sqrt(dop))*((4*ds.vh)/(ds.vv + ds.vh))
                # Concat 
                data_ds = xr.concat([data, ds], dim='field')
            except RasterioIOError as e:  # rasterio.errors.RasterioIOError
                print(f'Skipping coordinate {coordinate} for {date_slice}: {str(e)}')
                continue

    if save:
        data_ds.to_netcdf(f'data_{start_date}_{end_date}_{bands}.nc')  # Save xArray to cwd
        print('XArray Dataset saved to disk!')

    return data_ds
