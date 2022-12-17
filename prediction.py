# supress all warnings
import warnings 
warnings.filterwarnings('ignore')

# import modules
import pandas as pd
import numpy as np

# For serialization:
import joblib
import pickle


tableColNames1 = ["ici_3400_gar","ici_4200_gar","ici_5000_gar","ici_5800_gar","ici_6500_gar"]

for cols in tableColNames1:
    joblib_preds = joblib.load(f'Model_AutoArima_{cols}_Model.pkl').predict(n_periods=13)
    print(cols,' Prediction :', joblib_preds)