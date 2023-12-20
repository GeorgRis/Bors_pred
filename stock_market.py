import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import datetime

osebx = yf.Ticker('OSEBX.OL')

osebx = osebx.history(period='max')

del osebx['Stock Splits']
del osebx['Dividends']


osebx['Dag1'] = osebx['Close'].shift(-1)
osebx['morgendag'] = (osebx['Dag1'] > osebx['Close']).astype(int)       
osebx = osebx.loc['2019-01-01':].copy()

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = osebx.iloc[:-100]
test = osebx.iloc[-100:]

predictors = ['Close', 'Volume','Open','High','Low']


def predict(train, test, predictors, model):
    model.fit(train[predictors], train['morgendag'])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combine = pd.concat([test['morgendag'], preds], axis=1)
    return combine


def backtest(data, model, predictors, start=250, step=25):
    all_predict = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predict.append(predictions)        
    return pd.concat(all_predict)



horizons = [2, 5, 60 , 250]
new_predictors = []

for horizon in horizons:
    rolling_average = osebx.rolling(horizon).mean()
    ratio_column = f'Close_Ratio_{horizon}'
    osebx[ratio_column] = osebx['Close']/rolling_average['Close']
    
    trend_column = f'Trend_{horizon}'
    osebx[trend_column] = osebx.shift(1).rolling(horizon).sum()['morgendag']
    
    new_predictors += [ratio_column, trend_column]
    
osebx = osebx.dropna()
    
predictions = backtest(osebx, model, new_predictors)

predictions.to_csv('predictions_file.csv')

# Predict for future dates
future_dates = pd.date_range(start=osebx.index[-1], periods=50, freq='B')
future_predictions = pd.DataFrame(index=future_dates, columns=predictions.columns)

for date in future_dates:
    future_data = osebx.loc[:date].copy()
    future_predictions.loc[date] = predict(train, future_data, predictors, model).iloc[-1]

combined_predictions = pd.concat([predictions, future_predictions])

combined_predictions.to_csv('combined_predictions.csv')

print(predictions['Predictions'].value_counts())
print(precision_score(predictions['morgendag'], predictions['Predictions']))
print(predictions['morgendag'].value_counts()/predictions.shape[0])


print(osebx)