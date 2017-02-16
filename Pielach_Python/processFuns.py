# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:57:04 2016
@author: h0540603
"""
import pandas as pd
import numpy as np
import cv2
import os, glob
import exifread

def getImagesAsArray(csv_files, start = 0, stop = []):

    if not stop:
        stop = len(csv_files[start:])

    data = np.zeros(1)

    for f in csv_files[start:stop]:
        d =  pd.read_csv(f, sep = ';', header = None)
        d = d.dropna(axis = 1, how = 'all')
        d = d.as_matrix().flatten()
        data = np.concatenate((data, d))

    data = data[1:]
    return data

def csv2tiffs_scaled(csv_dir, tiff_dir, name, start = 0, stop = [],
                     q_low = 1.0, q_up = 98.0):
    '''
    start: the first processed image
    stop: the first NOT processed image (last processed + 1)
    q_low: qunatile for the lower limit for the inner scaling part
    q_up: quantile for the lower limit for the inner scaling part
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    os.chdir(csv_dir)
    csv_files = glob.glob('*.csv')
    if not stop:
        stop = len(csv_files[start:])

    if isinstance(start, str):
        start = csv_files.index(start)
    if isinstance(stop, str):
            stop = csv_files.index(stop)

    csv_selected = csv_files[start:stop]

    uint16_max = 65000.0
    raw_data = getImagesAsArray(csv_selected)
    maxV = np.percentile(raw_data, 99.9)
    minV = np.percentile(raw_data, 0.1)

    data = raw_data.copy()
    data[raw_data < minV] = minV
    data[raw_data > maxV] = maxV

    q_upper = np.percentile(data, q_up)
    q_lower = np.percentile(data, q_low)

    data_scaled = np.zeros(1)

    for f in csv_selected:
        d = pd.read_csv(f, sep = ';', header = None)
        d = d.dropna(axis = 1, how = 'all')
        img = d.as_matrix()
        img[img < minV] = minV
        img[img > maxV] = maxV

        # Get masks for the three parts: below the lower quantile, above the upper
        # quantile and in between
        low = img < q_lower
        high = img > q_upper
        middle = ~(low + high)

        section_low = 0.1
        section_high = 0.9

        img_scaled = np.zeros(img.shape)

        img_scaled[low] = (img[low] - minV)/(q_lower - minV) * uint16_max*section_low
        img_scaled[middle] = (img[middle] - q_lower)/(q_upper - q_lower) * uint16_max*(1 - section_low - (1 - section_high)) + uint16_max*section_low
        img_scaled[high] = (img[high] - q_upper)/(maxV - q_upper) * uint16_max*(1 - section_high) + uint16_max*section_high

        img_scaled = img_scaled.astype(np.uint16)
        cv2.imwrite(os.path.join(tiff_dir, os.path.splitext(f)[0] + '_times10.tif'), img_scaled)
        print('saving: %s' % f)

        img_scaled_flat = img_scaled.flatten()
        data_scaled = np.concatenate((data_scaled, img_scaled_flat))

    data_scaled = data_scaled[1:]
    fig, axs = plt.subplots(2, 1)
    axs[1].hist(data_scaled, bins = 50, range = (0, uint16_max))
    axs[0].hist(data, bins = np.arange(minV, maxV, 0.5))

    fig2, axs2 = plt.subplots(2, 1)
    axs2[1].imshow(img_scaled, cmap = 'Greys', vmin = 0, vmax = uint16_max)
    axs2[0].imshow(img, cmap = 'Greys', vmin = minV, vmax = maxV)

    log = pd.DataFrame([minV, q_lower, q_upper, maxV,
                         csv_files[start], csv_files[stop]], columns = [name],
            index = ['Minimum', 'Q_%i' % int(q_low), 'Q_%i' % int(q_up), 'Maximum', 'Start', 'Stop'])
    log.to_csv(os.path.join(tiff_dir, 'Log.csv'))
    log

def csv2tiff(f, tiff_dir, f2 = [],):
    '''
    Create tiff file from csv
    If f2 is not empty, the csv file is compared to the next file in the list and
    only converted in case that is unequal to the next file. This is necessary
    as images in the beginning and end of the ravi video are saved multiple times
    due to the use of the hot key function.
    f2: prior image against with f is compared.
    '''

    d = pd.read_csv(f, sep = ';', header = None)
    d = d.dropna(axis = 1, how = 'all')
    img = d.as_matrix()

    if f2:
        d2 = pd.read_csv(f2, sep = ';', header = None)
        d2 = d2.dropna(axis = 1, how = 'all')
        img2 = d2.as_matrix()

        if ~np.all(np.equal(img, img2)):
            img[img < 0] = 0
            img = np.array(img * 100, dtype = np.uint16)
            cv2.imwrite(os.path.join(tiff_dir, os.path.splitext(f)[0] + '.tiff'), img)
            print('saving: %s' % f)

        else:
            print('%s is a duplicate' % f)
    else:
        img[img < 0] = 0
        img = np.array(img * 100, dtype = np.uint16)
        cv2.imwrite(os.path.join(tiff_dir, os.path.splitext(f)[0] + '.tiff'), img)
        print('saving: %s' % f)

def removeDuplicateTiffs(image_files):
    '''
    When saving color tiff TIR imagery or csv files in Pi Connect using
    the hot key functions, duplicates have to be removed in the beginning and
    end of the video. Additionally for some frames the frames are duplicates
    of each other during the flight too.
    '''
    dupList = []
    for n in range(1, len(image_files)):
        f = image_files[n]
        f2 = image_files[n - 1]

        if f.endswith('.csv'):
            d = pd.read_csv(f, sep = ';', header = None)
            im = (d.dropna(axis = 1, how = 'all')
                   .as_matrix())
            d2 = pd.read_csv(f2, sep = ';', header = None)
            im2 = (d2.dropna(axis = 1, how = 'all')
                     .as_matrix())

        elif f.endswith('.tiff'):
            im = cv2.imread(f, -1)
            im2 = cv2.imread(f2, -1)

        if np.equal(im, im2).all():
            dupList.append(f)
    [os.remove(x) for x in dupList]
    print('removing: ', [os.path.split(x) for x in dupList])

def getMinMax(csv_files):
    ''' Get minimum and maximum value of all images in this chunk'''
    minV, maxV = [999, -999]
    for n in range(len(csv_files)):
        f = csv_files[n]
        d = pd.read_csv(f, sep = ';', header = None)
        d = d.dropna(axis = 1, how = 'all')
        img = d.as_matrix()
        if img.min() < minV: minV = img.min()
        if img.max() > maxV: maxV = img.max()
    print('Min: %0.1f, Max: %0.1f' % (minV, maxV))
    return minV, maxV

def getTIR_Dataframe(files):
    '''Create a dataframe from TIR image names and extract acquistion time from
    the file name.
    '''
    d = pd.DataFrame(files, columns = ['Names_TIR'])

    d = d.assign(date = d['Names_TIR'].apply(lambda x:os.path.splitext(x)[0].replace('Record_', '').split('_')[0]))
    d = d.assign(time = d['Names_TIR'].apply(lambda x:os.path.splitext(x)[0].replace('Record_', '').split('_')[1]))
    d = d.assign(time = d['time'].apply(lambda x: x.replace('-', ':')))
    d = d.assign(Time_TIR = d['date'] + ' ' +  d['time'])
    d = d.assign(Time_TIR = pd.to_datetime(d['Time_TIR'], format="%Y-%m-%d %H:%M:%S"))
    #dates = pd.to_datetime(dates, format="%Y-%d-%m %H:%M:%S")
    d = d.set_index('Time_TIR', drop = False)
    del d['time']
    del d['date']
    return d

def get_img_info(img_dir, save_info = False):
    'Read image acquisition time and name of the optical imagery'
    img = glob.glob(os.path.join(img_dir, '*.jpg'))
    img = [os.path.split(x)[1] for x in img]
    img_info = pd.DataFrame(img, columns = ['Names_RGB'])

    for i in img:
        f = open(os.path.join(img_dir, i), 'rb')
        tags = exifread.process_file(f)
        t = tags['EXIF DateTimeDigitized'].values
        t = t.split()
        t = ' '.join((t[0].replace(':', '-'), t[1]))
        img_info.loc[img_info['Names_RGB'] == i, 'Time_RGB'] = t
        print('Reading data for: ' +  i)


    dates = img_info['Time_RGB'].as_matrix()
    img_info = img_info.set_index(pd.DatetimeIndex(dates))
    if save_info:
        if not os.path.isdir(os.path.join(img_dir, 'IMG_info')):
            os.mkdir(os.path.join(img_dir, 'IMG_info'))
        img_info.to_csv(os.path.join(img_dir, 'IMG_info', 'IMG_info.csv'))
    return img_info

def get_gps_data(gps_file, offset, elevation_offset = 0):
    ''' Read the GPS data. The offset between optical imagery and GPS unit has
    to be known in order to match the correct timesteps. Negativ values mean
    that the GPS unit is ahead.
    '''

    def strip(text):
        try:
            return float(text.split(',', 1)[0])
        except AttributeError:
            return text

    gps_data = pd.read_csv(gps_file, sep = ';',
                           usecols = ['Latitude', 'Longitude',
                           'Elevation(m)', 'Time', 'Altitude(m)'],
                            converters={'Altitude(m)': strip})

    gps_data = gps_data.rename(columns = {'Altitude(m)': 'Altitude',
                                'Elevation(m)': 'Elevation'})

    gps_data.loc[:, 'Altitude'] = gps_data.loc[:, 'Altitude'] + elevation_offset
    gps_data.loc[:, 'Elevation'] = gps_data.loc[:, 'Elevation'] + elevation_offset
    dates = gps_data['Time']

    gps_data = gps_data.rename(columns = {'Time' : 'Time_GPS'})
    dates = pd.to_datetime(dates, format="%d.%m.%Y %H:%M:%S.%f")

    # shift gps data by offset
    delta = pd.to_timedelta(offset, unit = 's')
    date = dates + delta

    gps_data = gps_data.assign(Time_GPS_corrected = pd.DatetimeIndex(date).floor(freq = '1s').values)

    gps_data = gps_data.set_index(pd.DatetimeIndex(date))
    groups = gps_data.groupby(pd.TimeGrouper(freq = '1s')).first()

    return groups
