#!/bin/bash
wget 'https://www.dropbox.com/s/xijrm1a25q40wr9/model1.hdf5?dl=1' -O 'model1.hdf5' 
wget 'https://www.dropbox.com/s/k9lzryd06xmxx3w/model2.hdf5?dl=1' -O 'model2.hdf5' 
wget 'https://www.dropbox.com/s/dkxemlhnfv5iyw7/model3.hdf5?dl=1' -O 'model3.hdf5'  
python3 hw5_val_Ensemble.py $1 $2 $3 $4 
