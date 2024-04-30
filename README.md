# ticondagrova Advanced Research & Analytics
repo for code and data pushed by tARA

## Contents
The code contained in this repo is code used to complete the analysis done for our clients and other projects done.

 - [Folders](#folders) 
 - [Modules](#modules)
 - [Songrove Botanicals](#songrove-botanicals)


To see the full write ups, etc. see https://www.stellargrove.com/tara.  

[Home](#ticondagrova-advanced-research--analytics)

### Folders
1. <b>data</b>: <i>holds data for modules that are not entirely self contained.</i>
2. <b>notebooks</b>: <i>contains Jupyter Notebooks that run different analyses that are not a part of a work done for Songrove Botanicals.</i>
3. <b>Songrove Botanicals</b>: <i>work done for our agriculture client <b> Songrove Botanicals </b> - a top not agricultural research squad.</i>
4. <b>sqls</b>: <i>sql scripts used in different modules / being worked on for different things.</i>
5. <b>tests</b>: <i>unit test cases for functions and routines found within tara.</i>
<br>
### Modules
1. **distirbutions.py**: This module holds several different things:
    1. **TSP:**: *First it holds the distribution for the Two-Sided-Power Distribution as described by Van Dorp 2004.  <br>
    To use this function you call TSP and provide the low, middle and upper bounds as well as the parameter n.  See the research for more on what each mean.  <br>
    With the function **TSP** you can generate measures of central tendency using the TSP distribution.  It will return a dictionary with the expected value, variance, alpha & beta values needed to perform the cdf function as well as the p and q values generated. <br>      
    2. **createTSPSample:** *returns a two-columned data_frame that holds the sample from the uniform distribution to generate the TSP value, and then the TSP value itself. <br>
    For example, given the parameter list [low,mid,hi,n] = [11,15,22,2] and a size of 100, the **createTSPSample** function would return a DataFrame with two columns that contained values like [0.403533, 15.2231].  
        <br>
    Finally there is a helper functin for the TSP distribution, which is  **checkTSPParams**.  This is inteneded for the PERT simulator to come.  It simply returns "len not correct" if the length of the list is not equal to 4.*
    3. **generateTSP:** Finally, calling the function **generateTSP** generates the a singular value from the Two Sided Power (TSP) distribution. It returns the random uniform sample taken to determine what TSP value it corresponds to, and then also returns the value extracted from the TSP distribution, based on the random uniform sample taken.  Again, see Van Dorp 2004 for more details. 
    2. **sum_product**: this function finds the sum product of two arrays a, and b.  The function is literally return np.sum(a * b).  Error handling will come in a DevOps sprint later in the summer.
    3. **getDefaultParameters**: this function returns commonly used values for different distributions.  The distributions captured are: tsp, normal, poisson, binomial, bernoulli, and sample size.  The values are returned in a dictionary.
    4. **getDataFrameNames**: helper function to return different forms of DataFrame.  The list includes: dataframe, df, data-frame.  More to come later.
    5. **DaCountDeMonteCarlo**:  this is a monte carlo simulator of sorts.  What is meant to do is generate datasets for you.  
    There are a two main types of functions on this class: **sample_** and **create_** functions.  
    Functions that begin with **sample_** are functions that sample from the distribution given.  For example **sampleFromNormal(mean, std, sample)** returns a DataFrame with data that is from a normally distributed variable with average value of *mean*, standard deviation of *std*, and a sample from the [0,1] uniform distribution.  
    **create_** functions however are designed to return sets of data, with rows equal to the *sampleSize* parameter in each data set.  The data contained in the sample is of the distribution specified with the parameters given.  For example, **createUniformData(a, b, sampleSize, output="list")** would return a *list* of size *sampleSize*, which is from a uniformly distributed random variable with low values of *a* and high values of *b*.  This format is followed for each of the different **create_** functions in the DaCountDeMonteCarlo class.  **Note:** the class is usually abbreviated as dcmc when imported into a file.
2. **grovebot.py:**: this is the module that will contain any code done for robotics programming with the kids.  It's an empty shell of a file, as we haven't started construction yet, but will be filled out over time.
3. **hub.py**:  This file is the file that contains all the examples shown for tara's research portfoliio.  The idea behind this file is to have a file that contains all the work from tara in one place that you can call.  Typical structure is that there are a series of functions / procedures held within the different classes that correspond to the different projects done.  The functions are executed from a Jupyter Notebook, which walks the user through the process of building the model and seeing the output.  Classes currently include:
    1. **CreditScore:** class creates data to build a logistic regression model to determine the percentage chance that a customer with a set of given attributes will default on their credit card or not.
    2. **Titanic:** **IN PROGRESS**  this class takes the Titanic dataset from Kaggle and does analysis on it.  I was in the middle of working on this, when I was 
4. **meta.py**: work I was doing for a take home challenge. Some decent stuff in here, but somewhat specialized, so not sure how much it will mean to anyone.
5. **pll.py:** this file contains a few different functions and classes to gather, clean and analyze data pertaining to the Premiere Lacrosse League (PLL).  See below for a listing of the file's contents.
    1. **variables:** variables used throughout the module.  The idea behind this is to gather variables that are used throughout the entire module in one place, so if anything needs to be changed you are doing it in one place, not multiple places throughout the file.  Examples of some variables are: *data_location*: the folder in the repo were data is stored, *DB*: a dictionary that contains the parameters needed to connect to, in this case, a local instance of MSSQLServer.
    2. **helper functions:**: these functions are resuable functions like *get_database* which is a function that you feed the name of a file .csv file that you want to load into a DataFrame to do work on.
    3. **Data:** this class handles the processing of data to get it into a usable form.  This is where examples of ELT and ETL can be shown.  Much of the data that is in here is surrounding player statistics, attendance, and then some made up data to do analysis on.  The class also contains a function called *run_data_cleaner*.  This is a function that runs a series of smaller, more specialized processes.  The processes contain things like checking if the first rows of a DataFrame are all blank, signalling the need for the *skiprows* parameter.  Other functions include turning numbers stored as strings into decimals or integers, determine the number of null values in each column, look for character values in numeric columns, etc.
    4. **simulator:** this class has a series of functions and processes that generate simulated data for customer attributes and other things.  The idea behind this class is to show how, given rules, you can simulate out possible scenarios, then analyze said scenarios over and over again, gather what is hopefully a more accurate picture of the problem you are looking to analyze.
    5. **analysis:** this class shows off some of the heavier Data Science / Machine Learning techniques like clustering, classification, modelling and predictions.  ML gets involved by automating these individual analyses to create an automated analysis.
6. **stuffs.py:**  this file holds a bunch of constans like connection strings, data locations etc. It's a way to put everything that is commonly referenced in one place for maintenance, though some of the modules have their own variables set, so **stuffs.py** isn't used.


[Home](#ticondagrova-advanced-research--analytics)
### Data
There are 10 csv files corresponding to the different analyses done, all of which are listed below, with short descriptions of what is contained in them.
 1. **CropRotation**: *shows a set of data that has different crop rotations along with the phosphorus, nitrogen and potassium found in the soil as a result of the crop rotation.*
 2. **FertilizerTypes**: *this data set shows different crops, the fertilizer types and the yields associated with them.*
 3. **HarvestingMethods**: *this data set contains the results from a series of experiments that looks at how the plants were harvested.  Options contained: Hand Plot, Mechanical, Strip and Selective Harvesting Methods.  The weight of the plant in ounces is used as the measuring metric for how well / not well a particular method performed.*
 4. **IrrigationMethods**: *the data in this file describes four different irrigation methods: Drip, Flood, Sprinkler, and Furrow along with their yields and growth rates. The four methods are tested on both Yield and Growth Rate.*
 5. **PestControl**: *this data set shows the number of pests counted, 24 hours post treatment. The data is part the study using predators to deal with the pests as opposed to pesticides.*
 6. **PlantingTimes**: *planting times and their effects on different crops are measured here.  In this data set, three different crops: corn, wheat & greens are planted in three different parts of the growing season: early, middle and late.*
 7. **SeedVarieties**: *these data show a series of seeds types and the heights of the resulting plants that grew from them.*
 8. **SoilTypes**: *four different soil types are blinded and the resulting yields from the crops planted were measured in pounds.*
 9. **StorageMethods**:
10. **WeatherImpact**:

[Home](#ticondagrova-advanced-research--analytics)
### Notebooks
1. **ANOVA Template.ipynb:** This is a template that is used when performing ANOVA analyses.
2. **Crop Rotation.ipynb:** This notebook looks at the analysis done for Crop Rotation experiments.
3. **Fertilizer Type.ipynb:** Looking at the work we did with Fertilizers, use this notebook to walk through everything that we did.   
4. **Harvesting Methods.ipynb:** 
5. **Irrigation Methods.ipynb:**
6. **Pest Control.ipynb:**
7. **Planting Times.ipynb:**
8. **Seed Varieties.ipynb:**
9. **Soil Types.ipynb:**
10. **Storage Methods.ipynb:**
11. **Weather Impact.ipynb:**


