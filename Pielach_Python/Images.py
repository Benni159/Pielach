# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:11:52 2017

@author: bapperl
"""

#%% h in mm of preciptiation for transmission

import numpy as np
import matplotlib.pyplot as plt
        
    
Rv= 461.4               # spec. gas constant for water 
Temp_C=np.linspace(0,40,401)  
#Magnus formula to relate satuartion vapor pressure and temperature
esw = (10**(22.5518+(-2937.4/(Temp_C+273.15)))*(Temp_C+273.15)**(-4.9283))*10    # in mbar
H1 = 100/Rv/(Temp_C+273.15)*esw
Vh20_RH100 =  H1*1000  #mm/km
h_RH=RH*Vh20_RH100*distance*10**-3

plt.plot(Temp_C,h_RH)

plt.title('Precipitation as a function of temperature for RH = 100%')
# make axis labels
plt.xlabel('Temperature in °C')
plt.ylabel('h in mm/km H20')


#%% Atmosperic correction TIR mean, weghted with spectral power, weghted spectral and sensitiviy sensor triangle

#Atmosperic correction has to be done first

import numpy as np
import matplotlib.pyplot as plt

A=plt.plot(column_h[1:],PLWater_avg[1:], label='Artihmetic Mean')
plt.hold(True)
plt.plot(column_h[1:],PLWater_avg_weigthed[1:], label='Weighted Spectral Average')
plt.plot(column_h[1:],PLWater_avg_weighted_fuzzy[1:], label= 'Weighted +sensor sensitivity')
plt.ylim(0.8, 1)
plt.xlim(0,5)
plt.legend(loc='upper right', fontsize=8)
plt.title("Transmittance values for spectral band 7.5-13 micorns")
plt.xlabel("precipitable water h in mm")
plt.ylabel("athmosferictransmittance")

#%% This plot shows the resulting temperature difference to the tir image temperautre cuased by emissivity and transmissivity correction

import numpy as np
import matplotlib.pyplot as plt

ta=np.linspace(250,350,51)
temp=300
e=np.linspace(0.5,1,11)
transmissivity =1
diff_ta_temp=ta-temp

labels=[]
for i in range(0,len(e)):
    temp_surface=((temp**4/transmissivity-((1-e[i])*ta**4))/e[i])**0.25
    plt.plot(diff_ta_temp,temp_surface-temp)
    labels.append(r'e= %1.2f' % (e[i]))
    plt.hold(True)
    
plt.legend(labels, loc='lower left',fontsize=8, fancybox=True, shadow=True)
plt.xlabel('Delta T (Ta-T)')
plt.ylabel('Resulting Temperature Difference')
plt.title('Effects of Reflectance and Tempertue difference Resulting Temperature')

plt.show() 

#%% Plot data from weather station

plt.plot()
   




