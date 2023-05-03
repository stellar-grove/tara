
import os
import sys; sys.path.append("../")
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
from itertools import combinations
import statsmodels.stats.power as smp
import math

# ------ Begin Constants ----- #
PEEP = os.environ["USERNAME"]
DATADIR = f'C:/stellar-grove/tara/SongroveBotanicals/data/'
ascendingValues = ["a","asc","ascend","ascending"]
descendingValues = ["d","desc","descend","descending"]
ttestValues = ["ttest","t","t-test","student"]
DOMINANCE_COLS = ["Method 1", "Method 2", "Mean 1", "Mean 2", "TestStat", "p-value","pValSig"]
DECIMAL_PRECISION = 6
ALPHA_VALUE = 0.05
DEFAULT_POWER = 0.9
DEFAULT_TTEST_IND_RATIO = 1



# ------- Helper Functions -----------#
def updateLog(existing:list,new):
        existing.append(new)

def determinePower(effect_size, ObsCount, 
                   alpha:float=ALPHA_VALUE, ind:bool = False):
        if ind:
            power = smp.TTestIndPower().power(effect_size=effect_size,
                                              nobs=ObsCount,
                                              alpha=alpha)
        else:
            power = smp.TTestPower().power(effect_size=effect_size,
                                           nobs=ObsCount,
                                           alpha=alpha)
        return power

def determineSampleSize(effect_size, power:float=DEFAULT_POWER, 
                        alpha:float=ALPHA_VALUE, ind:bool = False,ratio:float=None):
        if ind:
            size = smp.TTestIndPower().solve_power(effect_size=effect_size,
                                                   nobs1=None,
                                                   power=power,
                                                   alpha=alpha,
                                                   ratio=DEFAULT_TTEST_IND_RATIO)
            size = math.ceil(size)
        else:
            size = smp.TTestPower().solve_power(effect_size=effect_size,
                                                nobs=None,
                                                power=power,
                                                alpha=alpha)
            size = math.ceil(size)
        return size

class Irrigation(object):    
    def __init__(self, log={"status":[]})->None:
        self.log = {"status":[['C0','class initiated']]} 
        self.data = {}

    def loadData(self) -> tuple:
        self.log["status"].append(["DL0","Begin Data Load"])
        DATAPATH = f"{DATADIR}IrrigationMethods.csv"
        df = pd.read_csv(DATAPATH)
        self.log["status"].append(['DL1','Data Loaded'])
        dfYield = df[df["Factor"]=="Yield"].pivot(index="TrialNumber",
                                                  columns='IrrigationType', 
                                                  values='FactorValue')
        
        dfGrowthRate = df[df["Factor"]=="Growth Rate"].pivot(index="TrialNumber",
                                                             columns='IrrigationType', 
                                                             values='FactorValue')
        self.data["Yield"] = dfYield
        self.data["GrowthRate"] = dfGrowthRate
        return dfYield, dfGrowthRate
    
    
    def calculateMeans(self, data:pd.DataFrame, sort_order:str=None):
        updateLog(self.log["status"],["CM0", "Means initiated"])
        means = pd.DataFrame(data.mean()).reset_index()
        means.columns = ["Method", "AverageYield"]
        updateLog(self.log["status"],["CM1", "Means calculated"])
        if sort_order != None:
            if sort_order.lower() in ascendingValues:
                means.sort_values(by=means.columns[1],
                                  ascending=True,
                                  inplace=True)
                
            if sort_order.lower() in descendingValues:
                means.sort_values(by=means.columns[1],
                                  ascending=False,
                                  inplace=True)
        self.data["means"] = means.set_index("Method")
        updateLog(self.log["status"],["CM2", "Means written"])
        return means

    def runANOVA(self,data:pd.DataFrame) -> tuple:
        updateLog(self.log["status"],["A0", "ANOVA initiated"])
        f_statistic, p_value = f_oneway(data["Drip"],
                                data["Flood"],
                                data["Furrow"],
                                data["Sprinkler"]
                                )
        f_statistic = round(f_statistic,6)
        p_value = round(p_value,6)
        updateLog(self.log["status"],["A1", "ANOVA calculated"])
        self.data["ANOVA"] = {"f_statistics":f_statistic,
                              "p_value":p_value}
        updateLog(self.log["status"],["A2", "ANOVA stats written"])
        return f_statistic, p_value

    def runDominance(self, data:pd.DataFrame,test:str="t-test")->tuple:
        if test.lower() in ttestValues:
            results = []
            cols = list(data.columns)
            combos = list(combinations(cols, 2))
            for i in range(0,len(combos)):
                A = combos[i][0]
                B = combos[i][1]
                stat, pvalue = ttest_ind(data[A], data[B])
                mean_A = data[A].mean()
                mean_B = data[B].mean()
                pvalsig = True if pvalue < ALPHA_VALUE else False
                interim = [A, 
                           B, 
                           mean_A, 
                           mean_B, 
                           round(stat,DECIMAL_PRECISION), 
                           round(pvalue,DECIMAL_PRECISION),pvalsig]
                
                results.append(interim) 
            results = pd.DataFrame(results, columns = DOMINANCE_COLS)
            results.sort_values(by="Method 1",
                                ascending=False,
                                inplace=True)
        return results
        
class CropRotation(object):
    log = {}
    
class Fertilizer(object):
    def __init__(self, log={"status":[]})->None:
        self.log = {"status":[['C0','class initiated']]} 
        self.data = {}

    def loadData(self) -> tuple:
        self.log["status"].append(["DL0","Begin Data Load"])
        DATAPATH = f"{DATADIR}FertilizerTypes.csv"
        df = pd.read_csv(DATAPATH)
        self.log["status"].append(['DL1','Data Loaded'])
        # dfYield = df[df["Factor"]=="Yield"].pivot(index="TrialNumber",
        #                                           columns='IrrigationType', 
        #                                           values='FactorValue')
        return df

class Harvesting(object):
    log = {}   

class PestControl(object):
    log = {} 

class PlantingTime(object):
    log = {} 

class SeedVarieties(object):
    log = {} 

class SoilTypes(object):
    log = {} 

class StorageTypes(object):
    log = {} 

class WeatherImpacts(object):
    log = {} 