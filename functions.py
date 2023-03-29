import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import pandas as pd

def get_sentinel_data(latlong,time_slice,assets):
    '''
    Returns VV and VH values for a given latitude and longitude 
    Attributes:
    latlong - A tuple with 2 elements - latitude and longitude
    time_slice - Timeframe for which the VV and VH values have to be extracted
    assets - A list of bands to be extracted
    '''

    #latlong=latlong.replace('(','').replace(')','').replace(' ','').split(',')
    box_size_deg = 0.0008 # Surrounding box in degrees, yields approximately 5x5 pixel region
    min_lon = float(latlong[1])-box_size_deg/2
    min_lat = float(latlong[0])-box_size_deg/2
    max_lon = float(latlong[1])+box_size_deg/2
    max_lat = float(latlong[0])+box_size_deg/2

    bbox_of_interest = (min_lon, min_lat, max_lon, max_lat)
    time_of_interest = time_slice

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    
    search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox_of_interest, datetime=time_of_interest)
    
    items = list(search.get_all_items())
    bands_of_interest = assets
    data = stac_load([items[0]], patch_url=pc.sign, bbox=bbox_of_interest).isel(time=0)
    vh = data["vh"].median().to_numpy()
    vv = data["vv"].median().to_numpy()
    
    return vh,vv


def convert_date(date_str):
    """
    Takes a string in the 'yyyy-mm-dd' format and returns a datetime.date object.
    """
    year, month, day = map(int, date_str.split('-'))
    return datetime.date(year, month, day)


def download_radar_data(latlong, start_date, end_date, assets = ['vh', 'vv'], window_month=1, window_day=0, save=True):
    '''
    Calls get_sentinel function to download data over a time period with snapshots occuring over given day/month windows.
    
    Parameters:
    latlong (list of tuples): Latitude and Longitude list of the locations to download data from.\n
    start_date (str): Date from which to begin downloading data in the 'yyyy-mm-dd' format, e.g. '2022-04-01'.\n
    end_date (str): Date on which to end the downloading in the 'yyyy-mm-dd' format, e.g. '2022-06-01'.\n
    assets (list): Assets to download, default is ['vh', 'vv']\n
    window_month (int): Windows in months between separate data downloads, only taken into account if window_day == 0.\n
    window_day (int): Windows in days between separate data downloads, if not default (0), it will take precedence over window_month input\n
    save (bool): If True it will automatically save the data to a csv file after finishing the download\n
    
    Returns:
    pandas.DataFrame: DataFrame with the downloaded assets
    '''


    # Define start and end dates
    start_date = convert_date(start_date)
    end_date = convert_date(end_date)

    # Define list to store data
    data = []

    # Iterate over range of dates
    while start_date <= end_date:
    
        if window_day == 0:
            next_date = start_date + relativedelta(months=+window_month)
        else:
            next_date = start_date + relativedelta(day=+window_day)
            
        time_slice = f'{start_date}/{next_date}'
        for coordinates in tqdm(latlong):
            vv, vh = get_sentinel_data(coordinates, time_slice, assets)
            data.append([time_slice, vv, vh])
        start_date = next_date

    # Convert list to DataFrame and save to file
    columns = ['date'] + assets
    data_vh_vv = pd.DataFrame(data, columns=columns)
    if save == True:
        data_vh_vv.to_csv(f'data_{start_date}_{end_date}.csv', index=False)
    return data_vh_vv
