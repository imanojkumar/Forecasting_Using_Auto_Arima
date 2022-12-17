# supress all warnings
import warnings 
warnings.filterwarnings('ignore')

# import modules
import pmdarima as pm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pmdarima import ARIMA as Arima, auto_arima as AutoArima
from mysql.connector import MySQLConnection, Error
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from pmdarima.arima import StepwiseContext


# For serialization:
import joblib
import pickle
import gc

#####################################################################################

# connect database
def connect():
    try:
        # conn = MySQLConnection(host='10.81.92.26', port=1436, user='root', password='Gr@phite#456$', database='IndexPrices')
        conn = MySQLConnection(host='localhost', port=3306, user='root', password='Walmart@13M', database='IndexPrices')
        if conn.is_connected():
            print('Connected to MySQL server')
    except Error as error:
        print(f'Error: {error}')
    return conn

#####################################################################################

# MySQL to Pandas
def SQlToPandas(table='indices'):
    
    cnx = connect()
    # Use the `read_sql` function to execute a SQL query and store the resulting data in a Pandas dataframe
    query = f"SELECT * FROM {table};"
    df = pd.read_sql(query, cnx)

    return df

#####################################################################################

# auto arima model
def AutoArimaModel(df, col):

    df1 = df[col].copy()
    
    # Train Test Split
    # minDate = df1.index.min()
    # maxDate = df1.index.max()
    # fixedDate = '2020-10-02'

    # train = df1['2009-11-06' : '2020-09-25']     # everything but cut-off upto 13 weeks before Jan 1st 2021 
    # test  = df1['2020-10-02' : '2028-12-28']     # 13 weeks before Jan 2020
    # train = train[train > 0.0]

    df1 = df1[df1 > 0.0]

    # Fit an ARIMA model to the time series
    with StepwiseContext(max_dur=15):
        model = pm.auto_arima(df1, start_p=0, start_q=0,
                            test='adf',       # use adftest to find optimal 'd'
                            max_p=3, max_q=3, # maximum p and q
                            m=52,              # frequency of series
                            d=1,           # let model determine 'd'
                            seasonal=True,   # Check Seasonality
                            start_P=0, start_Q=0,
                            max_P=3, max_Q=3, 
                            trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True, n_jobs=-1)

    return model

###############################################################################

if __name__ == '__main__':

    df = SQlToPandas()
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    df.set_index('date', inplace=True)


    for cols in df.columns.to_list():

        model = AutoArimaModel(df=df, col=cols)
        
        # Serialize with Pickle
        pkl = open(f'Model_AutoArima_{cols}_Model.pkl', 'wb')
        pickle.dump(model, pkl)
        pkl.close()
        
        print(f'Model_AutoArima_{cols}_Model.pkl file saved..')

        del model
        gc.collect()


