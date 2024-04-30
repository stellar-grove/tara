import os
import sys; sys.path.append("../")
import tara.stuffs as stuffs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



# ----------------------------------------------------
#   Helper Functions
# ----------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# split_data: data -> Data Frame that has the data you want to run the analysis on.
# split_data: features_list -> Holds a list of columns names where the first n items 
# are the names of the independent variables being used in the study.  The last entry in the list
# corresponds to the dependent variable that is being modelled.
def test_train(data:pd.DataFrame, features_list:list)->tuple:
    # Split data into training and validation sets
    # First determine the independent variables.
    X = data[features_list[:-1]]
    y = data[features_list[-1:]]
    # Next run the test / train split algorithm
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=33)
    # return the outputs of the train test split algorithm
    return X_train, X_val, y_train, y_val
# --------------------------------------------------------------------------------------------------------------
def function_narrative(d_var:list, d_coef:list, d_intercept:list, decimals:int):
    function_list = []
    for i in range(len(d_var)):
        var_name = d_var[i]
        var_coef = str(round(d_coef[i],decimals))
        intercept = str(round(d_intercept[0],decimals))
        function_list.append(var_coef + " * " + var_name)
    function_string = " ".join(function_list)
    function_string = f"f(x) = {function_string} {intercept}"
    return function_string

def confusion_matrix_output(y_train, predictions, labels:list) -> tuple:
    confusion = pd.DataFrame(metrics.confusion_matrix(y_train, 
                                                      predictions),
                                                      index=labels,
                                                      columns=labels)
    confusion_pct = confusion / sum(confusion.sum())
    return confusion, confusion_pct

def confusion_matrix_narrative(confusion_matrices:tuple):
    confusion_matrix = confusion_matrices[0]
    confusion_pct = confusion_matrices[1]

    total_right = confusion_matrix["No"]["No"] + confusion_matrix["Yes"]["Yes"]
    pct_right = confusion_pct["No"]["No"] + confusion_pct["Yes"]["Yes"]

    pred_right_but_wrong = confusion_matrix["Yes"]["No"]
    pct_right_but_wrong = confusion_pct["Yes"]["No"]
    pred_wrong_but_right = confusion_matrix["No"]["Yes"]
    pct_wrong_but_right = confusion_pct["No"]["Yes"]


    confusion_narrative = f"""Per the confusion matrix, the amount of items that were correctly predicted were {total_right} or {pct_right}.
    The amount of predictions were predicted to be Yes but were actually No was: {pred_right_but_wrong} or {pct_right_but_wrong}.
    The amount of predictions were predicted to be No but were actually Yes was: {pred_wrong_but_right} or {pct_wrong_but_right}"""

    return confusion_narrative

def BuildModel(split_data:list)->tuple:
    # Set variables for the data sets
    y_train = np.ravel(split_data[0])
    y_val = np.ravel(split_data[1])
    X_train = split_data[2]
    X_val = split_data[3]

    dict_model = {}
    dict_data = {}
    # Train model
    model = LogisticRegression()
    fitted_model = model.fit(X_train, y_train)
    dict_model["model"] = fitted_model
    dict_model["d_var"] = list(X_train.columns)
    dict_model["d_coef"] = [c for c in model.coef_[0]]
    dict_model["intercept"] = model.intercept_
    dict_model["model_words"] = function_narrative(dict_model["d_var"],
                                                dict_model["d_coef"],
                                                dict_model["intercept"],
                                                4)
    dict_model["accuracy_score"] = model.score(X_val,y_val)
    predictions = model.predict(X_train)
    dict_data["predictions"] = pd.DataFrame(predictions,columns=["Prediction"])
    confusion_matricies = confusion_matrix_output(y_train,
                                                dict_data["predictions"],
                                                ["No","Yes"])
    dict_model["confusion_matrix"] = confusion_matricies[0]
    dict_model["confusion_pct"] = confusion_matricies[1]
    dict_model["confusion_narrative"] = confusion_matrix_narrative(confusion_matricies)
    return dict_model, dict_data


class CreditScore(object):
    def __init__(self, log={"status":[]})->None:
        self.log = {"status":[['C0','Credit Score Class Initiated']]} 
        self.data = {}
        self.config = {}
        self.model = {}

    def loadData(self, test_size=0.2, fileName = "CreditScore.csv"):

        # This function grabs the data from the data folder in the repo and then
        # splits it into a few different pieces:
        #   1) the entire data set
        #   2) All the independent or X variables
        #   3) The variable being modeled or the Y variable.
        #   4) The different components of the test_train split: X_train, X_val, y_train, y_val
        # All these datasets are put into keys of the class level data dictionary.
        # The split data is put into a list, so when you call it it will have four elements,
        # each of which being one of the outputs of the test_train procedure.

        data_location = f"{stuffs.DATA_FOLDER}{fileName}"
        df = pd.read_csv(data_location)
        self.data["CreditScore"] = df
        self.data["X"] = df.iloc[:, -1:]
        self.data["y"] = df.iloc[:, :-1]
        X_train, X_val, y_train, y_val = train_test_split(self.data["X"], self.data["y"],
                                                          test_size=test_size,
                                                          random_state=33)
        split_data = [X_train, X_val, y_train, y_val]
        self.data["split_data"] = split_data
        

    def BuildModel(self):
        split_data = self.data["split_data"]
        model, data = BuildModel(split_data)
        for key in model.keys():
            self.model[key] = model[key]
        for key in data.keys():
            self.data[key] = data[key]

    def CreatePredictions(self):
        model = self.config["model"]
        predictions = model.predict(self.data["X_train"])
        return predictions
    
class Titanic(object):
    def __init__(self):
        self.data = {}
    # Define any class specific variables
    study = "Titanic"
    file_location = f'{stuffs.kaggleWD}{study}'
    def load_data(self)->tuple:
        df_test = pd.read_csv(f"{self.file_location}/test.csv")
        df_test["data_set"] = "test"
        df_train = pd.read_csv(f"{self.file_location}/train.csv")
        df_train["data_set"] = "train"
        self.data["test_set"] = df_test
        self.data["train_set"] = df_train
        return df_test,df_train

    def combine_data_sets(self):
        df_data = pd.concat([self.data["test_set"], self.data["train_set"]],ignore_index=True)
        return df_data

    def process_data(self):
        test, train = self.load_data()
        combined_data = self.combine_data_sets()
        return combined_data
    

