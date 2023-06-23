import os

DATA_FOLDER = "C:/stellar-grove/tara/data/"
DATA_FOLDER_SONGROVE = "C:/stellar-grove/tara/SongroveBotanicals/data/"


# ------ Begin Constants ----- #
peep = os.environ["USERNAME"]
robot = os.environ["COMPUTERNAME"]
homeD = f'{os.environ["HOMEDRIVE"]}{os.environ["HOMEPATH"]}'.replace('\\','/')

SGWD = f'{homeD}/Stellar Grove/'
bitsWD = f'{SGWD}bitstaemr - Documents/'
taraWD = f'{SGWD}ticondagrova - Documents/'

kaggleWD = f'{taraWD}Kaggle/'
server = f'{robot}\SQLEXPRESS'
DB_tara = {'servername': server,
        'database': 'tara',
        'driver': 'driver=SQL Server Native Client 11.0'
        ,'tgtSchema':'pll'
        ,'tgtTbl':'player_stats'
        ,'fileRoot':'C:/stellar-grove/tara/data/pll/'}