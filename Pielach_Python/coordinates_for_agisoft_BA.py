"""

Script to create a coordinate reference input file for Agisoft Photoscan from
two files: the GPS information from the UAV and the meta from the RGB imagery.
The aim is to get a dataset of harmonized time stamps.
"""

# Imports
import os
os.chdir('d:\Doktorat\Messungen\Pielach\Daten\Pielach\Auswertungen\Python')

import pandas as pd
import processFuns as funs



main_dir = 'd:\Doktorat\Messungen\Pielach\Daten\Pielach'
#Flight log_file
flight_table=pd.read_csv(os.path.join(main_dir, 'Messungen.csv'),
                         sep= ',')


for i in range(0,len(flight_table)):

    day=flight_table.iloc[i,0]
    time=flight_table.iloc[i,1]

#Folders

    # Static folders
    main_dir = 'd:\Doktorat\Messungen\Pielach\Daten\Pielach'
    gps_dir = os.path.join(main_dir, 'SD_Karte_LOG')
    offset_dir = os.path.join(main_dir, 'SD_Karte_LOG')



    # Directory of specific flight
    home = os.path.join(main_dir, day, time, 'Thermal', 'Bilder', 'PseudoRGB')
    img_dir = os.path.join(main_dir,day,time, 'Optisch')     # optical imagery

    name = home.split(os.sep)[-5] + '_' + home.split(os.sep)[-4]

    # Create GPS file name from folger structure
    gps_file = 'Track_' + name + '.csv'
    gps_file = os.path.join(gps_dir, gps_file)


#%% Create coordinates file



    # Set offset between camera and GPS unit. Check start of rotors and start of GPS
    offset_table = pd.read_csv(os.path.join(offset_dir, 'GPS_offset_Elevation.csv'),
                               sep = ';', index_col = 0)
    offset2optical = offset_table.loc[name, 'offset2optical']
    elevation_offset = offset_table.loc[name, 'elevation_offset']


    # Flag to decide whether to calculate img_info or not. Is takes some time and
    # thus might be avoided in case that it already exists
    calc_img_info = True
    if calc_img_info:
        img_info = funs.get_img_info(img_dir, save_info = True)
    else:
        img_info = pd.read_csv(os.path.join(img_dir, 'IMG_info', 'IMG_info.csv'),
                               usecols = ['Names_RGB', 'Time_RGB'])
        img_info = img_info.set_index(pd.DatetimeIndex(img_info['Time_RGB'].as_matrix()))

    gps_data = funs.get_gps_data(gps_file, offset2optical, elevation_offset)

    # Merge GPS data to images
    data = gps_data.merge(img_info, how = 'right', left_index = True, right_index = True)
    if not os.path.isdir(os.path.join(img_dir, 'Coordinates')):
                os.mkdir(os.path.join(img_dir, 'Coordinates'))
    data.to_pickle(os.path.join(img_dir, 'Coordinates', 'Coords_' + name + '.pkl'))
    data.to_csv(os.path.join(img_dir, 'Coordinates', 'Coords_' + name + '.csv'))
    print('Saved reference data for flight: ' + name)
