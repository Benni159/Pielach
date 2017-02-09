# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:25:42 2017

@author: bapperl
"""




import PhotoScan
import os
os.chdir('d:\Doktorat\Messungen\Pielach\Daten\Pielach\Auswertungen\Python')
import glob
import pandas as pd

    

#Define Chunks
c1 = PhotoScan.app.document.chunks[0]
c2 = PhotoScan.app.document.chunks[1]


# get Working Directory
a=c1.cameras[0]                         #Get first cemera frame
home_path=a.photo.path                  #Get Path of foto
b=len(home_path.split('/')[-1])+1    #get length of photo sting
home_path=home_path[:-b]                #eleimiate photo name

    

#load coordinates table

os.chdir(os.path.join(home_path, 'Coordinates'))
result = [i for i in glob.glob('*.{}'.format('csv'))]
                             
coordinates_table=pd.read_csv(os.path.join(home_path, 'Coordinates', result[0]),
                         sep= ',')    
    

#Merge Cameras

camsRGB=c1.cameras
cameraRGB=[x.label for x in camsRGB]
camsTIR=c2.cameras
cameraTIR=[x.label for x in camsTIR]

cameraRGB1=pd.DataFrame(cameraRGB, columns=['Names_RGB'])
cameraTIR1=pd.DataFrame(cameraTIR,columns=['TIR'])

cameras=cameraRGB1.merge(cameraTIR1,left_index=True,right_index=True)

#Merge cameras with coordintes table

coordinates_table_merged=coordinates_table.merge(cameras, on='Names_RGB')
     
coordinates_table_merged.to_csv((os.path.join(home_path, 'Coordinates', result[0])),sep=',',index=False)



