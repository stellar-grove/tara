
import os
import sys; sys.path.append("../")
import pandas as pd

# ------ Begin Constants ----- #
PEEP = os.environ["USERNAME"]
DATADIR = f'C:/stellar-grove/tara/data/'

class ExampleOne(object):
    def __init__(self, log={"status":[]})->None:
        self.log = {"status":[['C0','class initiated']]} 
    
    
    def loadData(self,FileName:str):
        self.log["status"].append(["DL0","Begin Data Load"])
        DATAFILENAME = "ExampleOne.csv"
        DATAPATH = f'{DATADIR}{DATAFILENAME}'
        useColumns = ["TrialNumber","Field","Value"]
        df = pd.read_csv(DATAPATH)
        self.log["status"].append(['DL1','Data Loaded'])
        df = df[useColumns].pivot(index=useColumns[0],columns=useColumns[1],values=useColumns[2])
        self.log["status"].append(['DL2','Data Transformed'])
        return df


class ExampleTwo(object):
    log = {}
    
    