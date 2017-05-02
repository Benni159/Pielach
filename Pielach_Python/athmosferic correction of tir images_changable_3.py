# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:42:00 2017

@author: bapperl

NECESSARY TO RUN THE FILE:
    
    CSV files
    Angle Files for Tiff images saved from AGISOFT to path see camera angle
    Log files GPS_offest

ATHMOSPEHRIC CORRECTION and emissivity correction

"""


#Packages#

import datetime
import numpy as np
import os
os.chdir('d:\Doktorat\Messungen\Pielach\Daten\Pielach\Auswertungen\Python')
import glob
import pandas as pd
import cv2
import pielach_funs as pfs
import math
import matplotlib.pyplot as plt
main_dir = 'd:\Doktorat\Messungen\Pielach\Daten\Pielach'



#%% 
'''INPUT VARIAbLES'''

#GET DATE

#ACTIVATE IF ONLY 1 RUN
#date = 'D20160826'
#time = 'T17'

#ACTIVATE IF LIST OF RUNS  
measurements = pd.read_csv(os.path.join(main_dir,'Messungen.csv'),header=0,sep=',')
#measurements=measurements.drop(7)
 
for q in range(8,len(measurements)):
    date = measurements.get_value(q,'Day')
    time = measurements.get_value(q,'Time')

    day = (date[-2:])
    month = (date[-4:-2])
    year = (date[1:5])
    
    if len(time) == 5:
        minute = time[3:5]
        hour = time[1:3]
        
        Datetime = datetime.datetime(int(year),int(month),int(day),int(hour),int(minute),0,0)
        
    else:
        Datetime = datetime.datetime(int(year),int(month),int(day),int(time[1:]),0,0)
    
    ### Define Directiories
    csv_dir = os.path.join(main_dir, date, time, 'Thermal', 'Bilder',)
    TIR_dir = os.path.join(csv_dir,'tiff_pseudocolor')
    GPS_offset_dir=os.path.join(main_dir,'SD_Karte_LOG')
    
    TIRlambda_min = 7.5       # microns
    TIRlambda_max = 13        #microns
    Temp = 290                # approximate Temperature of water surface in K
    
    
    #Resolution of Tir images
    x_res = 382
    y_res = 288
    #focal length
    f =  553
    deltah = 2               #2m offset wiel zwischen drohnenstartpunkt und wasser delta h=2m    : + fürDrohenstart höher als Zeilflaeche
    #flight height offest to be set if fix




      
    
    #%% load tables
    
    #load coordinates table
    
    os.chdir(os.path.join(main_dir,date,time,'Optisch\Coordinates'))
    result = [i for i in glob.glob('*.{}'.format('csv'))]
                       
    coordinates_table=pd.read_csv(os.path.join(main_dir,date,time,'Optisch\Coordinates', result[0]),
                             sep= ',')   
    
    #load camera angles (From Agisoft, made with the normal TIR images, with hihger contrast)
    camera_angle=pd.read_table(os.path.join(main_dir,'Auswertungen',(day+month+year),time,
                                            ('YawPitchRoll_EstimatesTIR_'+date+'_'+time+'.txt')),sep=',',header=1,skiprows=0)
    
    #Get image files
    csv_dir = os.path.join(main_dir, date, time, 'Thermal', 'Bilder',)
    os.chdir(csv_dir)
    csv_files = glob.glob('*.csv') 
    
    os.chdir(TIR_dir)
    TIR_files = glob.glob('*.tif') 
    
    #Get CsV list to correct
    start_image=TIR_files[1]
    start_image=start_image[:-12]+'.csv'
    final_image=TIR_files[-1]
    final_image=final_image[:-12]+'.csv'
    csv_files_to_correct=csv_files[csv_files.index(start_image):csv_files.index(final_image)]
    
                         
                                   
    offset_table = pd.read_csv(os.path.join(GPS_offset_dir, 'GPS_offset_Elevation.csv'),
                                   sep = ';', index_col = 0)
    elevation_offset = offset_table.loc[(date+'_'+time), 'elevation_offset'] 
                                        
    # Load passmann laromre table
    PLWater=pd.read_csv('d:\Doktorat\Messungen\Pielach\Daten\Pielach\Auswertungen\Other input data\Passmore Larmore Table_H20.csv',sep=';', header=0)
    PLco2=pd.read_csv('d:\Doktorat\Messungen\Pielach\Daten\Pielach\Auswertungen\Other input data\Passmore Larmore Table_CO2.csv',sep=';', header=0)
    PLWater_local=PLWater[(PLWater.Wavelength >=TIRlambda_min) & (PLWater.Wavelength <= TIRlambda_max)]
    
    
    #Load data from weatherstation
    Weatherstation = pd.read_csv('d:\Doktorat\Messungen\Pielach\Daten\wetterstation\Final Meteorological data\Pielach_t1.dat', sep=',', header=1, skiprows=0)
    Weatherstation=Weatherstation.drop(Weatherstation.index[[0,1]])
    Weatherstation=Weatherstation.set_index(pd.DatetimeIndex(Weatherstation['TIMESTAMP']))
    del Weatherstation['TIMESTAMP']
    Weatherstation=Weatherstation.astype(float)
    
        
    drift =pd.read_csv('d:\Doktorat\Messungen\Pielach\Daten\Pielach\Auswertungen\Athmosferic correction\Drift\linear gradient.csv', sep = ';', header = 0)
    
    #Get downwelling athomsferic transmittance from Low Tran
    L_down=pd.read_csv(os.path.join(main_dir,'Auswertungen','Athmosferic correction','Transmittance','Lowtran_Transmissivity_Path_to_space_Midaltitudesummer.csv'))
   
    
    
    #%% Get relevant data from tables^^
    
    if q > 2:
    
        RH = Weatherstation.loc[pd.to_datetime(Datetime),'v02humidity']/100
        ta = Weatherstation.loc[pd.to_datetime(Datetime),'v01temp']
    elif q == 1:
        RH = 0.52
        ta = 25.1
    elif q == 0:
        RH = 0.48
        ta = 27.2
    elif q == 2:
        RH = 0.65
        ta = 21.2
        
    
   
    #%%Define paths of otuput tables
        
    #create path:
    directory = os.path.join(csv_dir,'Pixelangle') 
    if not os.path.exists(directory):
        os.makedirs(directory)
              
    os.chdir(directory)
    angle_pseudocolor_dir = os.path.join(csv_dir,'Pixelangle\Pseudocolor') 
    if not os.path.exists(angle_pseudocolor_dir):
        os.makedirs(angle_pseudocolor_dir) 
       
    os.chdir(csv_dir)
    transmissivity_dir = os.path.join(csv_dir,'transmissivity') 
    if not os.path.exists(transmissivity_dir):
        os.makedirs(transmissivity_dir)
            
    os.chdir(csv_dir)
    temp_corr_dir = os.path.join(csv_dir,'temp_corr') 
    if not os.path.exists(temp_corr_dir):
        os.makedirs(temp_corr_dir)
        
    os.chdir(temp_corr_dir)
    temp_corr_just_weights_dir = os.path.join(temp_corr_dir,'just_radiance_weights') 
    if not os.path.exists(temp_corr_dir):
        os.makedirs(temp_corr_dir)    
     
    os.chdir(csv_dir)
    temp_corr_dir_drift = os.path.join(csv_dir,'temp_corr_plus_drift') 
    if not os.path.exists(temp_corr_dir_drift):
        os.makedirs(temp_corr_dir_drift)          
                
     



 #%% Calculate refraction index for water and weghted values
    
    h = 6.626*10**-34 #(J*s    =Planck constant)
    c = 299792458     # (speed of light m/s)    

    # Emissivity wehgthted for water
    refraction_trans = (2*math.pi*h*c**2/((PLWater_local['Wavelength'])*10**-6)**5)/(math.e**((0.0144)/(PLWater_local['Wavelength']*10**-6)/(Temp))-1)    #spektrale spez.Ausstahrlung 
                       
    
    #Calcualte weihgt for air tmerpareture wavelegth intensity ta
    fuzzy_weight = np.linspace(0,1,len(PLWater_local)/2)
    fuzzy_weight = np.append(fuzzy_weight,fuzzy_weight[::-1])
    weight_ta=(2*math.pi*h*c**2/((PLWater_local['Wavelength'])*10**-6)**5)/(math.e**((0.0144)/(PLWater_local['Wavelength']*10**-6)/(ta+273.15))-1)
    PLWater_avg_weighted_fuzzy_ta_Lup=np.average(PLWater_local,axis=0,weights=(weight_ta*fuzzy_weight))
    
    #calculated weighted atmosphere temp intensity
    weight_ta_Ldown=(2*math.pi*h*c**2/((L_down['wavelength_nm'])*10**-9)**5)/(math.e**((0.0144)/(L_down['wavelength_nm']*10**-9)/(ta+273.15))-1)
    L_down_transmit=np.average(L_down['transmission'],axis = 0, weights = weight_ta_Ldown)
    
      #%% Calculate data for every image      
    


    # Get downwelling transmissivity for mit-altitude summer from whole atmosphere averaged over the wavelength spectrum
       # Weighting is doen afterwards in transmittance definition
    
                                
                                        
                                        
    for m in range(0,len(TIR_files)):
        filename=TIR_files[m]  
    
        
        if coordinates_table['TIR'].str.contains(filename).any()==True:
        
            z_abs=coordinates_table.loc[coordinates_table['TIR'] == filename,'Altitude'].values
            z=z_abs-elevation_offset+deltah    
        
                                                    
                                            
        
            '''Resulitng variables'''
            # Calculate pixel dimension in m
            G= ((z*10**3)/(f-1))*10**-3
            
            
            # Distance matrix calculatingthe distance from the image center 
            surface_dist= np.zeros((y_res,x_res))
            
            for i in range(0,x_res):
                for j in range(0,y_res):
                    surface_dist[j][i]=np.sqrt(((np.absolute((x_res+1)/2-(i+1))-0.5)*G)**2+((np.absolute((y_res+1)/2-(j+1))-0.5)*G)**2)
                       
                    
                    
            #Hypothenuse of distance without rotation
            hypothenuse=np.sqrt(surface_dist**2+z**2)
            
            
            '''Get angle of TIR camera pixels'''
            x_angle=np.zeros((y_res,x_res,))
            y_angle=np.zeros((y_res,x_res,))
            xy_angle=np.zeros((y_res,x_res,))
                    
            for i in range(0,x_res):
               for j in range(0,y_res):
                   y_angle[j][i]=np.degrees(np.arctan(((np.absolute((y_res+1)/2-(j+1))-0.5)*G)/z))     #center point of pixel
                   x_angle[j][i]=np.degrees(np.arctan(((np.absolute((x_res+1)/2-(i+1))-0.5)*G)/z)) 
                   xy_angle[j][i]=np.degrees(np.arctan(((surface_dist[j,i]/z))))
                   
            del(i,j)       
                
              
           
        
            '''include pitch and roll in zenith angle'''
        
            if camera_angle['#Label'].str.contains(filename).any()==True:
    
                yaw=camera_angle.loc[camera_angle['#Label'] == filename,'Yaw_est'].values
                pitch=camera_angle.loc[camera_angle['#Label'] == filename,'Pitch_est'].values
                roll=camera_angle.loc[camera_angle['#Label'] == filename,'Roll_est'].values
                
                x_mulitplik_layer=np.zeros((y_res,x_res))
                x_mulitplik_layer[0:(y_res-1),0:(int((x_res)/2))]=-1
                x_mulitplik_layer[0:(y_res-1),(int((x_res-1)/2))+1:]=1               
                x_angleCorr=x_angle*x_mulitplik_layer+roll
                
                y_mulitplik_layer=np.zeros((y_res,x_res,))
                y_mulitplik_layer[ :int((y_res/2)),:]=-1
                y_mulitplik_layer[int((y_res/2)):,:]=1               
                y_angleCorr=y_angle*y_mulitplik_layer-pitch
                
                
                xy_angle_corr=np.zeros((y_res,x_res))
                xy_angle_corr=np.degrees((np.sqrt((z*np.tan(np.deg2rad(abs(x_angleCorr))))**2+
                        (z*np.tan(np.deg2rad(abs(y_angleCorr))))**2))/z)   
            
                xy_dist_corr=z/np.cos(np.deg2rad(xy_angle_corr))     
                       
                cv2.imwrite(os.path.join(csv_dir,'Pixelangle','Angle_'+filename),xy_angle_corr)
                cv2.imwrite(os.path.join(csv_dir,'Pixelangle','Path_'+filename),xy_dist_corr)
                
                # Turn interactive plotting off
                    
                #plt.ioff() 
               # fig = plt.figure()
                #plt.savefig(os.path.join(angle_pseudocolor_dir,'PseudocolorAngle'+filename))
                #plt.matshow(xy_angle_corr)
                #plt.close('all')
                               
                percent=round((m/len(TIR_files)*100),1)
                print (filename + ' written    '+str(percent) +' %' )
                
                
            #   CALCULATE emissivity    
            
                refraction_index = pfs.refractive_index(TIRlambda_min,TIRlambda_max,Temp)    
                emissivity_corr=np.zeros((y_res,x_res))
                
                for i in range(0,y_res):
                   for j in range(0,x_res):
                       emissivity_corr[i][j] = pfs.emissivity_angle(xy_angle_corr[i,j],refraction_index)
                       
                cv2.imwrite(os.path.join(csv_dir,'Pixelangle','Emissivity_'+filename),emissivity_corr)
                
                print('            Emissivity finished')
                
            #   CALCULATE transmissivity ath pathlengh z
            
                transmissivity_corr=np.zeros((y_res,x_res))
                
            
                for i in range(0,y_res):
                   for j in range(0,x_res):
                       dist=xy_dist_corr[i,j]
                       transmissivity_corr[i][j] = pfs.transmissivity(RH,ta,dist,TIRlambda_min,TIRlambda_max,PLWater_avg_weighted_fuzzy_ta_Lup,refraction_trans)
                       
                np.savetxt(os.path.join(csv_dir,'transmissivity','Transmissivity_weights_sensisitivy_triangle'+(filename[:-11]+'.csv')),transmissivity_corr,delimiter=';',fmt='%.4f')
                       
                print('            Transmissivity finished')
                
                            
            #   CALCULATE PATH RADIANCE
            
                Path_Radiance=(1-transmissivity_corr)*ta**4
                
            #'''Recalculate image temperature´'''
                
                  #get image csv
                csv_name=filename[:-12]+'.csv'
                csv_file=np.zeros((y_res,x_res))
                csv_file=pd.read_csv(os.path.join(csv_dir,csv_name),sep= ';',header=None) 
                csv_file=csv_file.dropna(axis=1, how='all')
                                         
            
                    #calculate corrected temp
                temp_CORR=((((csv_file+273.15)**4-Path_Radiance)/transmissivity_corr-((1-emissivity_corr)*(1-L_down_transmit)*(ta+273.15)**4))/emissivity_corr)**0.25
                temp_CORR_inC = temp_CORR-273.15
                    
                diff_temp=(temp_CORR-csv_file)-273.15
                
                
                temp_CORR_inC.to_csv(os.path.join(csv_dir,'temp_corr' ,'temp_corr_'+csv_name),sep=';',header=None, index=None, float_format='%.2f')
                diff_temp.to_csv(os.path.join(csv_dir,'temp_corr' ,'difftemp_'+csv_name),sep=';',header=None, index=None, float_format='%.2f')
                #cv2.imwrite(os.path.join(csv_dir,'temp_corr' ,'temp_corr'+filename), temp_CORR_inC)
                #cv2.imwrite(os.path.join(csv_dir,'temp_corr' ,'difftemp'+filename), diff_temp)
                
                
            #Add linear temperature dirft
                
                
                drift_vect=np.linspace((len(TIR_files)*abs(drift.get_value(q,'Gradient linear'))),0,num=(len(TIR_files)))
                temp_CORR_in_C_drift=temp_CORR_inC-drift_vect[m-1]
                temp_CORR_in_C_drift.to_csv(os.path.join(temp_corr_dir_drift ,'temp_corr_drift_corr_'+csv_name),sep=';',header=None, index=None, float_format='%.2f')
            
            
                #m=m+1
        
    pfs.csv2tiffs(os.path.join(csv_dir,'temp_corr'),os.path.join(csv_dir,'temp_corr' ))
    pfs.csv2tiffs(os.path.join(temp_corr_dir_drift),os.path.join(temp_corr_dir_drift ))
     
    
    
      
    
     
           
    
    
           