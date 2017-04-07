# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:09:49 2017

@author: bapperl
"""


import PhotoScan

#Cameraposoition und TIR übernehmen mit

c1 = PhotoScan.app.document.chunks[2]
c2 = PhotoScan.app.document.chunks[1]

for i in range (len(c1.cameras)):
    c2.cameras[i].transform = c1.cameras[i].transform

    #Übertragen der Coordinaten auf TIR chunk
cameraRGB=c1.cameras
RGB_loc=[x.reference.location for x in cameraRGB]
cameraTIR=c2.cameras
for x in range (0,int(len(cameraTIR))):
    cameraTIR[x].reference.location =RGB_loc[x]

#Model übernehmen
c2.model = c1.model.copy()
