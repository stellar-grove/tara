import os
import sys; sys.path.append("../")
import tara.stuffs as stuffs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# ----------------------------------------------------
#   Helper Functions
# ----------------------------------------------------

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


class CreditScore(object):
    def __init__(self, log={"status":[]})->None:
        self.log = {"status":[['C0','Credit Score Class Initiated']]} 
        self.data = {}
        self.config = {}
        self.model = {}

    def loadData(self):
        data_location = f"{stuffs.DATA_FOLDER}CreditScore.csv"
        df = pd.read_csv(data_location)
        self.data["CreditScore"] = df
        self.data["X"] = df[['credit_score', 'income', 'loan_amount']]
        self.data["y"] = df[['repayment_status']]
        return df
    
    def BuildModel(self):
        # Split data into training and validation sets
        X = self.data["X"]
        y = self.data["y"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=33)
        self.data["X_train"] = X_train
        self.data["X_val"] = X_val
        self.data["y_train"] = y_train
        self.data["y_val"] = y_val

        # Train model
        model = LogisticRegression()
        fitted_model = model.fit(X_train, y_train)
        self.model["model"] = fitted_model

        self.model["d_var"] = list(X.columns)
        self.model["d_coef"] = [c for c in model.coef_[0]]
        self.model["intercept"] = model.intercept_
        self.model["model_words"] = function_narrative(self.model["d_var"],
                                                       self.model["d_coef"],
                                                       self.model["intercept"],
                                                       4)
        self.model["accuracy_score"] = model.score
        predictions = model.predict(X_train)
        self.data["predictions"] = predictions
        confusion_matricies = confusion_matrix_output(self.data["y_train"],
                                                      self.data["predictions"],
                                                      ["No","Yes"])
        self.model["confusion_matrix"] = confusion_matricies[0]
        self.model["confusion_pct"] = confusion_matricies[1]
        self.model["confusion_narrative"] = confusion_matrix_narrative(confusion_matricies)


    def CreatePredictions(self):
        model = self.config["model"]
        predictions = model.predict(self.data["X_train"])
        return predictions
    
