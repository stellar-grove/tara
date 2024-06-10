# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 22:39:13 2023

@author: DanielKorpon
"""

# --------------------------------------------------------------

dk_repo = "C:/repo/bitstaemr";sg_repo = "C:/stellar-grove"
import sys;sys.path.append(sg_repo)
import tara.stuffs as stuffs
import pandas as pd
import matplotlib.pyplot as plt
#sys.path.append(dk_repo)



class Kaggle(object):
    
    def __init__(self):
        self.stuff = {'wd':stuffs.kaggleWD}
        self.data = {}

    def getData(self, dataFolder:str=None):
        if dataFolder == None:
            return
        if dataFolder != None:
            fldr = f'{stuffs.kaggleWD}{dataFolder}/'
            fileList = self.list_files(fldr)
            return fileList

class Udemy(object):

    def setExercise(self, exercise):
        self.exercise = exercise
        return self.exercise
    
    class QuantFinance(object):
        def __init__(self, exercise:str=None):
            self.stuff = {'wd':f'{stuffs.udemyWD}Python for Finance and Algorithmic Trading with QuantConnect/',
                          'data_dir':f'{stuffs.udemyWD}Python for Finance and Algorithmic Trading with QuantConnect/DATA/'}
            self.exercise = exercise
            self.data = {}


        def getData(self, file_location, file_type):
            if file_type in ["csv", ".csv", "comma seperated values", "comma", "commas"]:
                df = pd.read_csv(file_location)
                return df