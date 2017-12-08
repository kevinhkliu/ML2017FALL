#!/bin/bash
wget 'https://www.dropbox.com/s/akiz7jmilrm1dt6/GRU_model_nolabel0.hdf5?dl=1' -O 'GRU_model_nolabel0.hdf5' 
wget 'https://www.dropbox.com/s/06m8m0ngm6icywc/GRU_model_nolabel1.hdf5?dl=1' -O 'GRU_model_nolabel1.hdf5' 
wget 'https://www.dropbox.com/s/9iw0zcmg7pnb359/GRU_model_nolabel2.hdf5?dl=1' -O 'GRU_model_nolabel2.hdf5' 
python3 hw4_saveTestData.py $1  
python3 hw4_val_Ensemble.py $2  
