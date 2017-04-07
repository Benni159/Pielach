# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:19:54 2017

@author: bapperl
"""

# This file transmitts the estimated GPs location of the drone from RGB alignment to TIR (just for 2nd Methode)

import PhotoScan
import os
os.chdir('d:\Doktorat\Messungen\Pielach\Daten\Pielach\Auswertungen\Python')
import glob
import pandas as pd




#Define Chunks
c1 = PhotoScan.app.document.chunks[0]
c2 = PhotoScan.app.document.chunks[5]


# get Working Directory
a=c1.cameras[0]                         #Get first cemera frame
home_path=a.photo.path                  #Get Path of foto
b=len(home_path.split('/')[-1])+1       #get length of photo string
home_path=home_path[:-b]                #eliminate photo name



# load coordinates table


os.chdir(os.path.join(home_path, 'Coordinates'))
result = [i for i in glob.glob('*.{}'.format('csv'))]

coordinates_table=pd.read_csv(os.path.join(home_path, 'Coordinates', result[0]),
                         sep= ',')


cameraRGB=c1.cameras
RGB_loc=[x.reference.location for x in cameraRGB]
cameraTIR=c2.cameras

for x in range (0,int(len(cameraTIR))):
   cameraTIR[x].reference.location =RGB_loc[x]
