# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 12:36:19 2021

@author: Mitch
"""
#imports
import upsilon
import csv
from mldatabase import open_database
import sys
import numpy as np




#Load classification model from Upsilon - load outside loops as takes 10-15s to restart
rf_model = upsilon.load_rf_model()                           

#Open darabase
with open_database('/fred/oz054/lmc') as df:
    
#Define job array size (here we will split data across 100 cores)
    array_size = 100

#Split data identifier and allocate script to each core within system. NB: task id's start from 1
    args = sys.argv[1:]
    jobid = int(args[0])-1

    #Split data into 100 equal pieces and allocate to each core
    split_data = np.array_split(df.objid.values, array_size)[jobid]
    #############################################################################################
    
    #Create csv file and put in heading
    file_name = 'Upsilon_classification_objects_jobid_' + str(jobid) + '.csv' #Give unique file name for each job
    with open(file_name, mode ='a+', newline='') as class_file:
            class_file = csv.writer(class_file, delimiter=',',quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            class_file.writerow(['objid','label', 'probability', 'flag', 'amplitude', 'hl_amp_ratio', 
                                  'kurtosis', 'period', 'phase_cusum', 'phase_eta', 'phi21', 'phi31', 
                                  'quartile31', 'r21', 'r31', 'shapiro_w', 'skewness', 'slope_per10', 'slope_per90',
                                  'stetson', 'cusum', 'eta', 'n_points', 'period_SNR', 'period_log10FAP', 
                                  'period_uncertainty', 'weighted_mean', 'weighted_std'])
      
    
    for objid in split_data:
        #Filter all objects with objid matching key value
        obj = df[df.objid == objid]
        
        #Obtain column containing light curve values
        mags = obj.mag
              
        
        #Calculate mean and standard deviation of magnitudes
        mean_mag = mags.mean()
        std_mag = mags.std()
        
        #Below we perform clipping of data more than 3 stabdard deviations from the mean of the light curve as required for
        #ideal Upsilon performance.
        
        #Select mag readings within 3 standard deviations of the  mean
        obj.select(mags > (mean_mag - 3*std_mag)) #greater than 3 standard deviations below mean
        obj.select(mags < (mean_mag + 3*std_mag), mode='and') #less than 3 standard deviations above mean
        
        #obtain NumPy array of HJD's that satidy above selection
        date = obj.evaluate(obj.hjd, selection=True)
        
        #Obtain the magnitude values as a NumPy array with above selection
        mag = obj.evaluate(mags, selection=True)
        
        #Obtain the errors in the magnitude readings for the above selection
        err = obj.evaluate(obj.magerr, selection=True)
            
    
    
        #Extract features of light curve using upsilon - this is the most time consuming step+++
        e_features = upsilon.ExtractFeatures(date, mag, err)
        e_features.run()
        features = e_features.get_features()
    
    
        # Classify the light curve - this is the second most time consuming step
        label, probability, flag = upsilon.predict(rf_model, features)
        
        
        
        #Extract features of light curve
        amplitude = features['amplitude']
        hl_amp_ratio = features['hl_amp_ratio']
        kurtosis = features['kurtosis']
        period = features['period']
        phase_cusum = features['phase_cusum']
        phase_eta = features['phase_eta']
        phi21 = features['phi21']
        phi31 = features['phi31']
        quartile31 = features['quartile31']
        r21 = features['r21']
        r31 = features['r31']
        shapiro_w = features['shapiro_w']
        skewness = features['skewness']
        slope_per10 = features['slope_per10']
        slope_per90 = features['slope_per90']
        stetson_k = features['stetson_k']
        cusum = features['cusum']
        eta = features['eta']
        n_points = features['n_points']
        period_SNR = features['period_SNR']
        period_log10FAP = features['period_log10FAP']
        period_uncertainty = features['period_uncertainty']
        weighted_mean = features['weighted_mean']
        weighted_std = features['weighted_std']
        
        
             
        
        
        #Write objid, classification and extracted features to file
        with open(file_name, mode ='a+', newline='') as class_file:
            class_file = csv.writer(class_file, delimiter=',',quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            class_file.writerow([objid, label, probability, flag, amplitude, hl_amp_ratio, 
                                  kurtosis, period, phase_cusum, phase_eta, phi21, phi31, 
                                  quartile31, r21, r31, shapiro_w, skewness, slope_per10, slope_per90,
                                  stetson_k, cusum, eta, n_points, period_SNR, period_log10FAP, 
                                  period_uncertainty, weighted_mean, weighted_std])
            

