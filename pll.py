import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys;sys.path.append("./")
import sqlalchemy
import pyodbc as db 

season_list = [2021,2022,2023]
name_root = "C:/Users/DanielKorpon/Downloads/pll-player-stats-"

sqlExpressConnectionString = r'Server=localhost\SQLEXPRESS;Database=tara;Trusted_Connection=True;'
sqlExpressInstanceName = 'SQLEXPRESS'
sqlADMIN = 'AzureAD\DanielKorpon'
sqlFeaturesInstalled = 'SQLENGINE'
sqlInitialVersion = '16.0.1000.6, RTM'
sqlServerInstallLogFolder = 'C:\\Program Files\\Microsoft SQL Server\\160\\Setup Bootstrap\\Log\\20230530_130059'
sqlInstallationMediaFolder = 'C:\\SQL2022\\Express_ENU'
sqlInstallationResourcesFolder = 'C:\\Program Files\\Microsoft SQL Server\\160\\SSEI\\Resources'

class Data(object):
    def __init__(self, log={"status":[]})->None:
        self.log = {"status":[['C0','Credit Score Class Initiated']]} 
        self.data = {}
        self.config = {}
        self.model = {}


    def load_data(self):
        df = pd.DataFrame()
        for season in season_list:
            file_location = f'{name_root}{season}.csv'
            pll_playerStats = pd.read_csv(file_location)
            pll_playerStats["season"] = season
            df = pd.concat([pll_playerStats,df],ignore_index=True)
        last_col = df.iloc[:, -1]
        df = pd.concat([last_col, df.iloc[:, :-1]], axis=1)
        return df

    def create_db_connection(self)->tuple:
        cnxn = "connection"; engine = "engine"
        return cnxn, engine

    def kill_db_connection(self, cnxn):
        cnxn.quit()

    def load_to_database(self, data_frame:pd.DataFrame(), table:str, schema:str='tara', chunksize:int=250):
        cnxn = db.connect('DRIVER={SQL Server};SERVER='+sqlExpressInstanceName+';DATABASE='+'tara'+';Trusted_Connection=True')
        engine = sqlalchemy.create_engine("mysql+mysqldb://Starfighter533\SQLEXPRESS/tara")
        data_frame.to_sql(table,schema,con=engine,if_exists='append',index=False, chunksize=chunksize)
        records = data_frame.shape[0]
        return records