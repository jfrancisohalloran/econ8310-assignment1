import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

train_data = pd.read_csv(
    'assignment_data_train.csv',
    parse_dates=['Timestamp'],
    index_col='Timestamp'
)

train_data = train_data.asfreq('h')

model = ExponentialSmoothing(
    train_data['trips'],        
    trend='add',                     
    seasonal='add',                  
    seasonal_periods=24*7            
)

modelFit = model.fit()

test_data = pd.read_csv(
    'assignment_data_test.csv',
    parse_dates=['Timestamp'],
    index_col='Timestamp'
)

test_data = test_data.asfreq('h')

steps_to_forecast = len(test_data)
pred = modelFit.forecast(steps=steps_to_forecast)

pred = pred.values
