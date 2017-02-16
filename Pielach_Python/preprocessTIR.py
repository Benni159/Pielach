# -*- coding: utf-8 -*-
"""
This script preprocesses thermal images for its use in Agisoft Photoscan and saves them to a subfolder tiff_pseudocolor
"""

import glob, os
import pandas as pd
import numpy as np
import cv2
import processFuns as funs
import matplotlib.pyplot as plt
import shutil

day = 'D20160826'
time = 'T14'


# Directories

main_dir = 'd:\Doktorat\Messungen\Pielach\Daten\Pielach'
offset_dir = os.path.join(main_dir, 'SD_Karte_LOG')

home = os.path.join(main_dir, day, time, 'Thermal', 'Bilder',)
csv_dir = home

#csv_rename_dir = os.path.join(home, 'Thermal', 'csv_rename')


#camera_dir = os.path.join(home, 'Agisoft', 'Cameras')
#tiff_color_dir = os.path.join(home, 'Thermal', 'tiff_color')
#tiff_color_rename_dir = os.path.join(home, 'Thermal', 'tiff_color_rename')
tiff_pseudocolor_dir = os.path.join(csv_dir,'tiff_pseudocolor')
if not os.path.exists(tiff_pseudocolor_dir):
    os.mkdir(tiff_pseudocolor_dir)

os.chdir(csv_dir)
csv_files = glob.glob('*.csv')
#tiff_pseudocolor_files = glob.glob(os.path.join(tiff_pseudocolor_dir, '*.tiff'))
#tiff_color_files = glob.glob(os.path.join(tiff_color_dir, '*.tiff'))

name = home.split(os.sep)[-4] + '_' + home.split(os.sep)[-3]

#%% Get min and max values of all frames

#minVal, maxVal = funs.getMinMax(csv_files)

#%% Get all images as data array

#all_data = funs.getImagesAsArray(csv_files)

#%% Remove dupicates from color tiffs (exported as tiff from Pi Connect)
#files = tiff_color_files
files = csv_files
funs.removeDuplicateTiffs(files)

#%% Load flight data (GPS and RGB imagery) from pickle and combine optical and
# thermal imagery

# Select either of them:
#os.chdir(tiff_color_dir)
#files = glob.glob('*.tiff')
#rename_dir = tiff_color_rename_dir

os.chdir(csv_dir)
files = glob.glob('*.csv')
rename_dir = csv_dir

##%%
#tir_data = funs.getTIR_Dataframe(files)
#
flight_data = pd.read_csv(os.path.join(main_dir,day,time, 'Optisch', 'Coordinates', ('Coords_%s.csv'
                                          % name)))
#
#
## Attention. Take not first image as the first has a time gap because of the
## removal of duplicates
#offset_table = pd.read_csv(os.path.join(offset_dir, 'GPS_offset_Elevation.csv'),
#                           sep = ';', index_col = 0)
#rgb_match = offset_table.loc[name, 'RGB_match']
#if files[0].endswith('.csv'):
#    tir_match = offset_table.loc[name, 'TIR_match_csv']
#elif files[0].endswith('.tiff'):
#    tir_match = offset_table.loc[name, 'TIR_match_tiff']
#
#rgb = flight_data.loc[flight_data['Names_RGB'] == rgb_match, 'Time_RGB']
#tir = tir_data.loc[tir_data['Names_TIR'] == tir_match, 'Time_TIR']
#
#rgb_time = pd.Timestamp(rgb.values[0])
#tir_time  = pd.Timestamp(tir.values[0])
#
#tir2optical = pd.Timedelta(rgb_time - tir_time)
#tir_data = tir_data.assign(Time_TIR_corrected = tir_data['Time_TIR'] + tir2optical)
#tir_data = tir_data.set_index('Time_TIR_corrected', drop = False)
#
#data = flight_data.merge(tir_data, how = 'outer', left_index = True, right_index = True)
#
## Create new names for tiff corresponding to its RGB counterpart
#if files[0].endswith('.csv'):
#    data = data.assign(Names_TIR_renamed = 'TIR' + data.loc[:, 'Names_RGB'].str[3:-4] + '.csv')
#    data = data.assign(Names_TIR_renamed_from_csv = 'TIR' + data.loc[:, 'Names_RGB'].str[3:-4] + '.tiff')
#if files[0].endswith('.tiff'):
#    data = data.assign(Names_TIR_renamed = 'TIR' + data.loc[:, 'Names_RGB'].str[3:-4] + '.tiff')
#data = data.loc[data['Names_TIR'].notnull(), :]
#
#data = data.loc[data['Names_RGB'].notnull(), :]
#
## Where no RGB image is available, the name of the TIR is also not defined.
## Thus I label them explicitly.'
##rgb_null = data.loc[data['Names_TIR_renamed'].isnull(), :]
##rgb_null = rgb_null.assign(Num = np.arange(len(rgb_null)))
##rgb_null['Names_TIR_renamed'] = 'TIR_noRGB_' + rgb_null['Num'].map(str) + '.tiff'
##del rgb_null['Num']
##
##data.loc[rgb_null.index, 'Names_TIR_renamed'] = rgb_null['Names_TIR_renamed']
##del rgb_null
#
#RGB_coordinates = pd.read_csv(os.path.join(camera_dir, 'Cameras_' + name + '.txt'),
#                              skiprows = 1, usecols = ['#Label', 'X_est',
#                                                       'Y_est', 'Z_est',
#                                                       'Yaw_est', 'Pitch_est',
#                                                       'Roll_est'])
#RGB_coordinates = RGB_coordinates.rename(columns = {'#Label' : 'Names_RGB'})
#data = data.merge(RGB_coordinates, how = 'left', on = 'Names_RGB')
#
#if not os.path.isdir(os.path.join(rename_dir, 'Coordinates')):
#            os.mkdir(os.path.join(rename_dir, 'Coordinates'))
#
#data.to_pickle(os.path.join(rename_dir, 'Coordinates', 'Coords_TIR_' + name + '.pkl'))
#data.to_csv(os.path.join(rename_dir, 'Coordinates', 'Coords_TIR_' + name + '.csv'))

#%% Rename TIR images to have same number code as RGB imagery
#for i, row in data.iterrows():
##    newname = os.path.splitext(row['Names_RGB'][3:])[0]
##    newname = 'TIR' + newname + '.tiff'
#    print(row['Names_TIR_renamed'])
##    shutil.copy2(os.path.join(tiff_color_dir, row['Names_TIR']),
##                 os.path.join(tiff_color_rename_dir, row['Names_TIR_renamed']))
#    shutil.copy2(os.path.join(csv_dir, row['Names_TIR']),
#             os.path.join(csv_rename_dir, row['Names_TIR_renamed']))

#%% Create scaled tiffs from csv and save it as 16-bit image


start = flight_data.at[1,'TIR']
start = start.replace('_times10.tif','.csv')
stop = flight_data.at[len(flight_data)-1,'TIR']
stop = stop.replace('_times10.tif','.csv')


q_low = 1.0
q_up = 98.0

funs.csv2tiffs_scaled(csv_dir, tiff_pseudocolor_dir, name,
                      start = start, stop = stop, q_low = q_low, q_up = q_up)

#%% Create tiffs from csv (10 x csv value)  and save it as 16-bit image

# Loop over all csv but the last and eliminate duplicates in the beginning and end
#for n in range(1, len(csv_files)):
#    funs.csv2tiff(csv_files[n], tiff_pseudocolor_dir, csv_files[n - 1])
#funs.csv2tiff(csv_files[0], tiff_pseudocolor_dir)
