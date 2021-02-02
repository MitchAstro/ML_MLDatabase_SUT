# Classification and plotting of stellar light curves using machine learning classifier UPSiLoN #

This repo contains two scripts.  The first script classifies stellar light curves using the machine learning stellar classification tool [UPSiLoN](https://github.com/dwkim78/upsilon).  The second can be used to filter and plot stellar light curves meeting certain desired criteria identified by UPSiLoN in the first script. Both scripts use the [MLDatabase](https://github.com/1313e/MLDatabase).  


## Upsilon_classification_objects_MLDatabase_parrallel_processing.py ##
This script depends on the machine learning stellar classification tool [UPSiLoN](https://github.com/dwkim78/upsilon).  This script also uses an "embarrassingly parallel" job array.  When running on a supercomputer, the following shell script may be of assistance:

```
#!/bin/bash 

#SBATCH -J ARRAY                    # Flag submission as Job Array
#SBATCH --array=1-100               # Number of cores you want to allocate to job
#SBATCH --ntasks=1                  # Number of tasks per core (i.e one script)
#SBATCH --cpus-per-task=2           # 1 core per task
#SBATCH --mem-per-cpu=25GB          # RAM required for each individual core 
#SBATCH --time=40:00:00             # [d-]hh:mm:ss, processing time per core


#SBATCH --job-name=JobArray_%J_%a    # blank_slurm_project
#SBATCH --output=output_JobArray_%J_%a.out        # stdout.%J
#SBATCH --error=stderr_JobArray_%J_%a.err        # stderr.%J


source activate my_env              #Activate python environment

#Identify your script run as a Slurm Job Array
python Upsilon_classification_objects_MLDatabase_parrallel_processing.py $SLURM_ARRAY_TASK_ID
```

A csv file containing the object ID and the all the features of the light curve identified by UPSiLoN is the end product of this script.  For more information regarding the calculated features, please see the UPSiLoN [README](https://github.com/dwkim78/upsilon/blob/master/README.md).


## Graph_light_curves_MLDatabase.py ##

This script identifies objects meeting certain parameters identified by UPSiLoN which have been stored in a csv file by `Upsilon_classification_objects_MLDatabase_parrallel_processing.py` and plots there stellar light curve.  The parameters used to identify objects can be altered by modifying the boolean mask `filter_char = (upsilon_class_df["period"] < 0.25) & (upsilon_class_df["period"] > 0.00) & (upsilon_class_df["flag"] == 0)`.  

The first light curve produced for each stellar object is a simple plot of time and light intensity.  The second is a folded light curve which depends on the light curve period identified by UPSiLoN.  


