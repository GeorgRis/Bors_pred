import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

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
    preds = model.predict(test[predictors])
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

predictions = backtest(osebx, model, predictors)



print(predictions['Predictions'].value_counts())
print(precision_score(predictions['morgendag'], predictions['Predictions']))
print(predictions['morgendag'].value_counts()/predictions.shape[0])


print(osebx)