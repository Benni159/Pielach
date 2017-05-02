# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:30:26 2017

@author: bapperl

"""

''' Find colsest arguments'''
def argfind(array, predicate):
        for i in range(array.shape[0]):
            if predicate(array[i]):
                return i
        return False

def find_nearest_above(array, value):
    return argfind(array, lambda x: x > value)




def make_video (image_dir):

    import cv2
    import os
    import glob

    os.chdir(image_dir)
    image_files = glob.glob('*.tif')



    img1 = cv2.imread(image_files[1])


    height , width , layers =  img1.shape

    video = cv2.VideoWriter('video.avi',-1,1,(width,height))


    for i in range(len(image_files)):

        video.write(cv2.imread(image_files[i]))
        print(i/len(image_files*100)+ '%')

    cv2.destroyAllWindows()
    video.release()


def refractive_index (lambda_min,lambda_max,Temp): #in microns and °K        CALCULATES THE AVERAGE REFRACTIVE INDEX OVER WAVELENGTH RANGE

    import math
    import pandas as pd

    h = 6.626*10**-34 #(J*s    =Planck constant)
    c = 299792458     # (speed of light m/s)


    refraction=pd.read_csv('d:\Doktorat\Messungen\Pielach\Daten\Pielach\Auswertungen\Other input data\water_refractive index_wavelngth_at 25°C.txt',sep= ',')
    refraction ['Ml']=(2*math.pi*h*c**2/((refraction['wl'])*10**-6)**5)/(math.e**((0.0144)/(refraction['wl']*10**-6)/(Temp))-1)     #spektrale spez.Ausstahrlung

    refraction=refraction[(refraction['wl'] >=lambda_min) & (refraction['wl'] <= lambda_max)]   #filter Values in wavelength

    #Calcualte weghted refaction
    refractive_index=sum(refraction.n*refraction.Ml)/sum(refraction.Ml)
    return (refractive_index)

    del(h,c)


def emissivity_angle (incident_angle,refractive_index):     # get the emissivity out of the incident angle

    import numpy as np

    xi = np.deg2rad(incident_angle)                    # for detailed description look up masuda et al. 1988
    xi_refracted = np.arcsin(np.sin(xi)*(1/refractive_index))
    n = refractive_index

    #calculate complex reflectances
    gnormal=-((n*np.cos(xi)-np.cos(xi_refracted))/((n*np.cos(xi)+np.cos(xi_refracted))))
    gparallel=((np.cos(xi)-n*np.cos(xi_refracted))/((np.cos(xi)+n*np.cos(xi_refracted))))

    #calculate complex refrecative index
    rho = ((abs(gnormal))**2+abs(gparallel)**2)/2

    emissivity=float(1-rho)
    #print(emissivity)
    return emissivity


def transmissivity (RH,ta,distance,lambda_min,lambda_max,PLWater_avg_weighted_fuzzy,refraction):

    import numpy as np
    #import pandas as pd
    import math

    ta_inK=ta+273.15

    #constants
    h = 6.626*10**-34 #(J*s    =Planck constant)
    c = 299792458     # (speed of light m/s)


    #Calculate the height h of prectipitable water according to Mininka 2016 and parsih (NASA, 1977)
    Rv= 461.4               # spec. gas constant for water
    #Temp_C=np.linspace(0,40,401)
    #Magnus formula to relate satuartion vapor pressure and temperature
    esw = (10**(22.5518+(-2937.4/(ta_inK)))*(ta_inK)**(-4.9283))*10    # in mbar
    H1 = 100/Rv/(ta_inK)*esw
    Vh20_RH100 =  H1*1000  #mm/km
    h_RH=RH*Vh20_RH100*distance*10**-3

    #load this tables to be faster (load this file in original file to be faster)
        #PLWater=pd.read_csv('d:\Doktorat\Messungen\Pielach\Daten\Pielach\Auswertungen\Other input data\Passmore Larmore Table_H20.csv',sep=';', header=0)
        #PLco2=pd.read_csv('d:\Doktorat\Messungen\Pielach\Daten\Pielach\Auswertungen\Other input data\Passmore Larmore Table_CO2.csv',sep=';', header=0)
    
        # filter wavelengths
        #PLWater_local=PLWater[(PLWater.Wavelength >=lambda_min) & (PLWater.Wavelength <= lambda_max)]
        
        #PLco2=PLco2[(PLco2.Wavelength >=lambda_min) & (PLco2.Wavelength <= lambda_max)]
    
        #Arthimetic mean over wavelength spektrum
        #PLWater_avg=np.mean(PLWater)
        
        
        ##ACTIVATE IF JUST WEIGTHED OVER WVAELENGTH: Weihted Average over wl spetrum with  P instensites
        #refraction =(2*math.pi*h*c**2/((PLWater_local['Wavelength'])*10**-6)**5)/(math.e**((0.0144)/(PLWater_local['Wavelength']*10**-6)/(Temp+273.15)-1))     #spektrale spez.Ausstahrlung
        #PLWater_avg_weigthed=np.average(PLWater_local,axis=0, weights=refraction)
        #PLWater_avg_weighted_fuzzy=PLWater_avg_weigthed
        
        #ACTIVATE IF WEIGTHED OVER WVAELENGTH AND CAMERA SENSITIVTYWeighted average combined with traingular sensitivity of camera over the spectrum
        #fuzzy_weight = np.linspace(0,1,len(PLWater_local)/2)
       #fuzzy_weight=np.append(fuzzy_weight,fuzzy_weight[::-1])
        #PLWater_avg_weighted_fuzzy=np.average(PLWater_local,axis=0,weights=(refraction*fuzzy_weight))

    # get Specific value
    column_h=np.array([0,0.2,0.5,1,2,5,10,20,50])
    
    #for accelartion of the process delete comlun from column_h not needed
    z=4 #files from rigth colmun o be ignored
    column_h=column_h[:-z]
    PLWater_avg_weighted_fuzzy = PLWater_avg_weighted_fuzzy[:-z]
    
    idx_up=find_nearest_above(column_h, h_RH)
    idx_low=idx_up-1
    Transmittance=(PLWater_avg_weighted_fuzzy[idx_up]-(column_h[idx_up]-h_RH)/(column_h [idx_up]-column_h[idx_low])* \
            abs(PLWater_avg_weighted_fuzzy[idx_up]-PLWater_avg_weighted_fuzzy[idx_low]))

    return Transmittance

    del(h,c,esw,H1,Vh20_RH100,h_RH,fuzzy_weight,idx_up,idx_low)


def csv2tiffs(csv_dir, tiff_dir, start = 0, stop = []):
    '''
    Create tiff file from csv
    This function generates tiff images by multiplying csv temperature values 
    by 10. 
    '''
    
    import numpy as np
    import pandas as pd
    import os
    import glob
    import cv2
    
    os.chdir(csv_dir)
    csv_files = glob.glob('*.csv')
    
    for f in csv_files:
        d = pd.read_csv(f, sep = ';', header = None)
        d = d.dropna(axis = 1, how = 'all')
        img = d.as_matrix()
        
        img[img < 0] = 0
        img = np.array(img * 100, dtype = np.uint16)
        cv2.imwrite(os.path.join(tiff_dir, os.path.splitext(f)[0] + '.tiff'), img)
        print('saving: %s' % f)
        
        
        
        



def my_legend(axis = None):    # In line labelling
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage

    if axis == None:
        axis = plt.gca()

    N = 32
    Nlines = len(axis.lines)
    print (Nlines)

    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    # the 'point of presence' matrix
    pop = np.zeros((Nlines, N, N), dtype=np.float)    

    for l in range(Nlines):
        # get xy data and scale it to the NxN squares
        xy = axis.lines[l].get_xydata()
        xy = (xy - [xmin,ymin]) / ([xmax-xmin, ymax-ymin]) * N
        xy = xy.astype(np.int32)
        # mask stuff outside plot        
        mask = (xy[:,0] >= 0) & (xy[:,0] < N) & (xy[:,1] >= 0) & (xy[:,1] < N)
        xy = xy[mask]
        # add to pop
        for p in xy:
            pop[l][tuple(p)] = 1.0

    # find whitespace, nice place for labels
    ws = 1.0 - (np.sum(pop, axis=0) > 0) * 1.0 
    # don't use the borders
    ws[:,0]   = 0
    ws[:,N-1] = 0
    ws[0,:]   = 0  
    ws[N-1,:] = 0  

    # blur the pop's
    for l in range(Nlines):
        pop[l] = ndimage.gaussian_filter(pop[l], sigma=N/5)

    for l in range(Nlines):
        # positive weights for current line, negative weight for others....
        w = -0.3 * np.ones(Nlines, dtype=np.float)
        w[l] = 0.5

        # calculate a field         
        p = ws + np.sum(w[:, np.newaxis, np.newaxis] * pop, axis=0)
        plt.figure()
        plt.imshow(p, interpolation='nearest')
        plt.title(axis.lines[l].get_label())

        pos = np.argmax(p)  # note, argmax flattens the array first 
        best_x, best_y =  (pos / N, pos % N) 
        x = xmin + (xmax-xmin) * best_x / N       
        y = ymin + (ymax-ymin) * best_y / N       


        axis.text(x, y, axis.lines[l].get_label(), 
                  horizontalalignment='center',
                  verticalalignment='center')


