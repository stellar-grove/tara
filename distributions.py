import numpy as np
import scipy.stats as stats
import random
import pandas as pd
import sys; sys.path.append("../")

def sum_product(a,b):

    return np.sum(a * b)

class Helpers(object):
    
    def __init__(self,config={}) -> None:
        self.config = config
        self.stats = {"error_details": []}
        self.data = {}

    def getDefaultParameters():
            """
            This routine provides a dictionary with all the default parameters for all distributions contained in this 
            module.
            """
            dict_config = {
                "sampleSize":100,
                "tsp":{
                        "name":"tsp",
                        "low":13,
                        "mid":18,
                        "hi":25,
                        "n":2
                        },
                "normal":{
                        "name":"normal",
                        "mean":3,
                        "std":1.4
                    },
                "poisson":{
                            "name":"poisson",
                            "mu":4
                        },
                "binomial":{
                            "name":"binomial",
                            "n":5,
                            "p":0.4
                            },
                "bernoulli":{
                            "name":"bernoulli",
                            "p":0.271
                            }
                    
            }
            return dict_config
    
    def getDataFrameNames():
        """
        This function provides the different types of names 
        that a data frame could be listed to.
        """
        lst = ["dataframe", "df", "data-frame"]
        return lst
    
    def sum_product(a,b):
        return np.sum(a * b)

class basic(object):
   def __init__(self,config={}) -> None:
        self.config = config

   def sum_product(a,b):
            return np.sum(a * b)

class TwoSidedPower(object):

    def moments(self, LowBound, Middle , UpperBound, n):
        '''
        This functions provides the details of a given TSP distribution.  For example, you provide the parameters and it 
        will return: E(x), Var, alpha, beta, p & q.  These parameters are calculated using van Dorps paper. 

        params: LowBound, Middle, UpperBound, n

        results: the values associated with the TSP, mean, variance, etc.
        
        '''
        # Set any upfront variables
        params = [LowBound, Middle, UpperBound]
        weights = [float(1/6), float(4/6), float(1/6)]
        #value1 = (2-2^(0.5))/4
        # Calculate values
        expected_value = sum_product(np.array(params), np.array(weights))
        variance = ((UpperBound - LowBound)**2) / 36
        alpha = (expected_value - LowBound) / (UpperBound - LowBound)
        alpha2 = variance / (UpperBound - LowBound)**2
        beta = ((alpha*(1-alpha)) / alpha2) - 1
        p = alpha * beta
        q = (1 - alpha) * beta
        # Write to the payload dictionary
        payload = {}
        payload["expected_value"] = expected_value
        payload["variance"] = variance
        payload["alpha"] = alpha
        payload["beta"] = beta
        payload["p"] = p
        payload["q"] = q
        return payload

    # parameters list takes the values [Low, Mid, Hi, n]
    def createTSPSample(self, parameterList:list, size):
        """
        This function creates a sample of data that is distributed according to the TSP that has a list of
        parameters passed in the function.

        params: Low, Middle, High, n
        return: sample set of data distributed according to TSP(L,M,H,n)
        """

        listSample = np.random.uniform(0, 1, size)
        listValues = [self.moments(parameterList, sample) for sample in listSample]
        listCombined = [listSample, listValues]
        dfSample = pd.DataFrame(listCombined).T
        dfSample.columns = ["randomSampleValue", "GeneratedTSPValue"]
        return dfSample

    def generateTSP(self, parametersList: list, sample):
        """
        This function returns a singular value of a TSP distribution according to the 
        parameters list.  For it example it will generate a number like 14, given a value
        from 0 -1, i.e. p.

        params: parameterList -> [low, middle, hi, n], sample -> value ranging from 0 - 1
        return: value between low and high
        """
        LowBound = float(parametersList[0])
        Mid = float(parametersList[1])
        HighBound = float(parametersList[2])
        n = parametersList[3]
        x_value = sample
        FM = (Mid - LowBound) / (HighBound - LowBound)
        BoundRange = HighBound - LowBound
        lower_value = pow(x_value * BoundRange * pow(Mid - LowBound, n-1),(1/n))
        upper_value = pow((1 - x_value) * BoundRange * pow(HighBound - Mid, n-1),(1/n))
        FYL = LowBound + lower_value
        FYU = HighBound - upper_value
        if x_value < FM: 
            generated_value = FYL
        else: generated_value = FYU
        return generated_value

    def checkTSPParams(parametersList: list):
        """
        This is a helper function to ensure that the correct number of parameters is being passed.

        params: parametersList
        return: error message "len not correct" on errors.
        """
        if len(parametersList) != 4: return "length not correct"

    def getDefaultParameters():
            """
            This routine provides a dictionary with all the default parameters for all distributions contained in this 
            module.
            """
            dict_config = {
                "sampleSize":100,
                "tsp":{
                        "name":"tsp",
                        "low":13,
                        "mid":18,
                        "hi":25,
                        "n":2
                        },
                "normal":{
                        "name":"normal",
                        "mean":3,
                        "std":1.4
                    },
                "poisson":{
                            "name":"poisson",
                            "mu":4
                        },
                "binomial":{
                            "name":"binomial",
                            "n":5,
                            "p":0.4
                            },
                "bernoulli":{
                            "name":"bernoulli",
                            "p":0.271
                            }
                    
            }
            return dict_config

    def getDataFrameNames():
        """
        This function provides the different types of names 
        that a data frame could be listed to.
        """
        lst = ["dataframe", "df", "data-frame"]
        return lst

class DaCountDeMonteCarlo(object):
    """
    Da Count De Monte Carlo:
    This class contains functions and routines to perform different types of simulations. It uses a lot of pandas, 
    numpy, scipy and others, including UDF's contained within the functions.

    Functions:
        1.)

    Routines:
        1.) setParameters
        2.) generateSingleSample
        3.) Sampling: Normal, Poisson, Bernoulli, Gamma, Exponential, Beta, (TSP)
        4.) Creating (generating datasets): Poisson, Uniform, Normal, Gamma, Exponential
                                            Binomial, Bernoulli, Beta, (TSP)
    """

    def __init__(self,config={}) -> None:
        self.config = Helpers.getDefaultParameters()
        self.stats = {"error_details": []}
        self.data = {}

    def setParameters(self,dict_update):
        self.config.update(dict_update)

    def generateSingleSample(self, dict_distribution):
        dfOutput = pd.DataFrame(self.createUniformData(0, 1, self.config["sampleSize"]), columns=["uniSample"])
        if dict_distribution["distributionName"].lower() in ["tsp","twosidedpower","two-sided-power"]:
            listParameters = [dict_distribution["distributionParameters"]["low"],
                                dict_distribution["distributionParameters"]["mid"],
                                dict_distribution["distributionParameters"]["high"],
                                dict_distribution["distributionParameters"]["n"]
                                ]
            print(listParameters)
            dfOutput.loc[:,"TSP"] = dfOutput["uniSample"].apply(lambda x: TwoSidedPower.generateTSP(listParameters, x))
        if dict_distribution["distributionName"].lower() == "normal":
            listParameters = [dict_distribution["distributionParameters"]["mean"],dict_distribution["distributionParameters"]["std"]]
            dfOutput.loc[:,"Normal"] = dfOutput["uniSample"].apply(lambda x: self.sampleFromNormal(listParameters[0], listParameters[1], x))
        if dict_distribution["distributionName"].lower() == "poisson":
            listParameters = [dict_distribution["distributionParameters"]["mu"]]
            dfOutput.loc[:,"Poisson"] = dfOutput["uniSample"].apply(lambda x: self.sampleFromPoisson(listParameters[0], x))
        return dfOutput

#------------------------------------------------------------------------------------------------------------------------------
#   Functions to sample data from a distribution.
#   These functions generate the value of a distribution, given a percentile.  This is different 
#------------------------------------------------------------------------------------------------------------------------------

    def sampleFromNormal(self, mean, std, sample):
        z = stats.norm.ppf(sample)
        sampledValue = mean + (z * std)
        return sampledValue
    
    def sampleFromPoisson(self, mu, sample):
        value = stats.poisson.ppf(sample, mu)
        return value

    def sampleFromBernoulli(self, sample, p, loc):
        sampledValue = stats.bernoulli.ppf(sample, p, loc)
        return sampledValue
    
    def sampleFromBinomial(self, sample, n, p, loc):
        sampledValue = stats.binom.ppf(sample, n, p, loc)
        return sampledValue
    
    def sampleFromGamma(self, sample, alpha, loc, scale):
        sampledValue = stats.gamma.ppf(sample, alpha, loc, scale)
        return sampledValue
    
    def sampleFromExponential(self, sample, loc, scale):
        sampledValue = stats.expon.ppf(sample, loc, scale)
        return sampledValue
    
    def sampleFromBeta(self, sample, a, b, loc, scale):
        sampledValue = stats.beta.ppf(self, sample, a, b, loc, scale)
        return sampledValue
    

#------------------------------------------------------------------------------------------------------------------------------
#   Functions to create data
#   The functions below are intended to be used as helper functions that generated data needed to run analyses.  The class
#   contains these functions because ultimately the class is devoted to monte carlo simulations, and needing to generate random
#   samples from a distribution are needed.
#------------------------------------------------------------------------------------------------------------------------------

    def createPoissonData(self, mu, sampleSize, output = "list"):
        lst = stats.poisson.rvs(mu,sampleSize)
        if output.lower() in ["list"]: lst
        if output.lower() in Helpers.getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    def createUniformData(a, b, sampleSize, output = "list"):
        lst = np.random.uniform(a, b, sampleSize)
        if output.lower() in ["list"]: lst
        if output.lower() in Helpers.getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    def createNormalData(self, mean,std,size,output = "list"):
        lst = stats.norm.rvs(mean,std,size)
        if output.lower() in ["list"]: lst
        if output.lower() in Helpers.getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    def createGammaData(self, alpha, size, output = "list"):
        lst = stats.gamma.rvs(alpha,size)
        if output.lower() in ["list"]: lst
        if output.lower() in Helpers.getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    # This requires that you pass the scale as equal to 1/lambda
    def createExponentialData(self, scale,location,size, output = "list"):
        lst = stats.expon.rvs(scale=(scale),loc=location,size=size)
        if output.lower() in ["list"]: lst
        if output.lower() in Helpers.getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    def createBinomialData(self, n, p, size, output = "list"):
        lst = stats.binom.rvs(n,p,size)
        if output.lower() in ["list"]: lst
        if output.lower() in Helpers.getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    def createBernoulliData(self, p, loc, size, output = "list"):
        lst = stats.bernoulli.rvs(p, loc ,size)
        if output.lower() in ["list"]: lst
        if output.lower() in Helpers.getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    def createBetaData(self, a, b, loc, scale, size, output = "list"):
        lst = stats.beta.rvs(a, b, loc, scale, size)
        if output.lower() in ["list"]: lst
        if output.lower() in Helpers.getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst