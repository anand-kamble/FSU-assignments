import h5py
import glob
import numpy as np
from typing import List
import re

    
class Dataset:
    def __init__(self,filename, directory = None) -> None:
        self.filename:str = filename
        self.directory = directory
        self.files = None
    
    
    def searchFiles(self) -> bool:
        if self.directory is None:
            print("No directory specified, cannot load files.")
            exit(1)
        else:
            self.files = glob.glob(self.directory + "/" + self.filename)
            self.files.sort(key=self.__sortFiles)
            return True
    
    def loadDataset(self):
        
    
    def __sortFiles(self, file:str):
        regex_filter = re.compile(self.filename)
        sort_key = file.replace(regex_filter.search(file).group(),'')
        sort_key = sort_key.replace(self.directory + "/", "")
        return int(sort_key)
    
                      
def main():
        
    with h5py.File('dataset/sedov_hdf5_plt_cnt_0000', 'r') as f:
        # List all the groups in the HDF5 file
        print("Groups in the HDF5 file:")
        for group in f.keys():
            print(group)
            group = f['dens']
            print("\nShape of the array:", group.shape)
        
if __name__ == "__main__":
    main()