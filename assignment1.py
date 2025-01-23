import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

train_data = pd.read_csv(
    'assignment_data_train.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

model = ExponentialSmoothing(
    train_data['trip_count'],        
    trend='add',                     
    seasonal='add',                  
    seasonal_periods=24*7            
)

modelFit = model.fit()

test_data = pd.read_csv(
    'assignment_data_test.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

pred = modelFit.forecast(steps=len(test_data))

pred = pred.values
