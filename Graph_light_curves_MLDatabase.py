# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:54:28 2021

@author: Mitch
"""
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import TimeSeries
from astropy.time import Time
from astropy import  units as u
import numpy as np
from mldatabase import open_database



counter = 0
#read in csv file containing upsilon classification and feautures of each object in MLD dataset
upsilon_class_df = pd.read_csv('Upsilon_classification_objects.csv')

#create boolean mask to filter objects with desired characteristics
filter_char = (upsilon_class_df["period"] < 0.25) & (upsilon_class_df["period"] > 0.00) & (upsilon_class_df["flag"] == 0) 

#Step 2 - apply boolean mask to dataframe to filter objects with desired characteristics
upsilon_class_df = upsilon_class_df[filter_char]


#Open darabase
with open_database('/fred/oz054/lmc') as df:

    #Loop takes desired objects which have been filted and stored in upsilon_class_df, and produces light curve plots
    for i in upsilon_class_df['objid']:
        
        #Get object with desired object id
        obj = df[df.objid == i] 
        
        
        #Extract column containing light curve
        mags = obj.mag
        
        #Calculate mean and standard deviation of light curve magnitudes
        mean_mag = mags.mean()
        std_mag = mags.std()
        
         #Select mag readings within 3 standard deviations of the  mean
        obj.select(mags > (mean_mag - 3*std_mag)) #greater than 3 standard deviations below mean
        obj.select(mags < (mean_mag + 3*std_mag), mode='and') #less than 3 standard deviations above mean
        
        #obtain NumPy array of HJD's that satidy above selection
        date = obj.evaluate(obj.hjd, selection=True)
        
        #Obtain the magnitude values as a NumPy array with above selection
        mag = obj.evaluate(mags, selection=True)

        #Obtain maximum magnitude and time of maximum magnitude (used later for epcoh time in folded light curve)
    
        max_mag = np.max(mag)
        max_time = date[mag == max_mag]
    
    
        #define period from upsilon and convert to units of time
        row_index = upsilon_class_df[upsilon_class_df['objid']== i].index[0] #get row number of object
        period = upsilon_class_df.loc[row_index, 'period'] #get periof of object based on upsilon
        period = period * u.second
    
           
        #Create time object from hjd data for star
        t = Time(date, format = 'mjd', scale = 'ut1')   
    
        #Create time series object using time object 't'
        ts = TimeSeries(time = t) #If scale = 'utc', gives: WARNING: ErfaWarning: ERFA function "utctai" yielded 561 of "dubious year (Note 3)" [astropy._erfa.core]
    
        #Plot light curve for star
        fig_1 = plt.figure()
        ts['mag'] = mag
        title = "Light curve for object " + str(i)
        plt.title(title)
        plt.plot(ts.time.jd, mag, 'k.', markersize=1)
        plt.xlabel("Julian Date")
        plt.ylabel(("Mag"))
        plt.gca().invert_yaxis()
        plt.show()
        name_file = str(i) +'.png'
        
        #Uncomment line below to save file in eps format
        fig_1.savefig(name_file, format = 'png') 
        
        #Close figure to save memory
        plt.close(fig_1)
        
        #Create folded light curve using TimeSeries in Astropy
        fig_2 = plt.figure()
        ts_folded = ts.fold(period = period) #epoch_time = ts.time[y == max_mag].ymdhms)
        title = "Folded light curve for object " + str(i)
        plt.title(title)
        plt.plot(ts_folded.time.jd, ts_folded['mag'], 'k.', markersize=1)
        plt.xlabel("Time (days)")
        plt.ylabel(("Mag"))
        plt.gca().invert_yaxis()
        plt.show()
        
        name_file = 'folded' + str(i) +'.png'
        
        #Uncomment line below to save file in eps format
        fig_2.savefig(name_file, format = 'png') 
        
        #Close figure to save memory
        plt.close(fig_2)
        
        counter += 1
    
        
print('Number of objects meeting requirements:')
print(counter)
    
    

    