import h5py
import numpy as np

with h5py.File('dataset/sedov_hdf5_plt_cnt_0000', 'r') as f:
    # List all the groups in the HDF5 file
    print("Groups in the HDF5 file:")
    for group in f.keys():
        print(group)
        group = f['dens']
        print("\nShape of the array:", group.shape)
    
    
class Dataset:
    def __init__(self,filename, directory = None) -> None:
        self.filename = filename
        self.directory = directory
    
    def LoadFiles(self) -> bool:
        if self.directory is None:
            print("No directory specified, cannot load files.")
            exit(1)
        else:
                     

