import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

data = get_rdataset('AirPassengers', 'datasets').data
data['time'] = pd.date_range(start='1949-01-01', periods=len(data), freq='M')
data.set_index('time', inplace=True)
ts = data['value']

print("Первые строки данных:\n", ts.head())
print("\nИнформация о данных:\n")
print(ts.info())
print("\nПропущенные значения:", ts.isna().sum())

plt.figure(figsize=(10, 5))
plt.plot(ts, label='Число пассажиров')
plt.title('Временной ряд AirPassengers')
plt.xlabel('Год')
plt.ylabel('Пассажиры (тыс.)')
plt.legend()
plt.show()

decomposition = seasonal_decompose(ts, model='additive', period=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(ts, label='Исходный ряд')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Тренд')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Сезонность')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Остатки')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

adf_result = adfuller(ts)
print('\nТест Дики-Фуллера:')
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
print('Критические значения:', adf_result[4])

decomposition_mult = seasonal_decompose(ts, model='multiplicative', period=12)

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(ts, label='Исходный ряд')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition_mult.trend, label='Тренд')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition_mult.seasonal, label='Сезонность')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition_mult.resid, label='Остатки')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

train = ts[:'1957-12']
test = ts['1958-01':]

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def theil_u(y_true, y_pred):
    naive = y_true.shift(1).fillna(y_true.iloc[0])
    error_model = np.sqrt(np.mean((y_pred - y_true)**2))
    error_naive = np.sqrt(np.mean((naive - y_true)**2))
    return error_model / error_naive

metrics = {'Model': [], 'MAE': [], 'RMSE': [], 'SMAPE': [], 'R2': [], 'Theil_U': []}
arima_model = auto_arima(train, seasonal=False, trace=False)
arima_fit = arima_model.fit(train)
arima_pred = arima_fit.predict(n_periods=len(test))
arima_pred = pd.Series(arima_pred, index=test.index)

metrics['Model'].append('ARIMA')
metrics['MAE'].append(mean_absolute_error(test, arima_pred))
metrics['RMSE'].append(np.sqrt(mean_squared_error(test, arima_pred)))
metrics['SMAPE'].append(smape(test, arima_pred))
metrics['R2'].append(r2_score(test, arima_pred))
metrics['Theil_U'].append(theil_u(test, arima_pred))

sarima_model = auto_arima(train, seasonal=True, m=12, trace=False)
sarima_fit = sarima_model.fit(train)
sarima_pred = sarima_fit.predict(n_periods=len(test))
sarima_pred = pd.Series(sarima_pred, index=test.index)

metrics['Model'].append('SARIMA')
metrics['MAE'].append(mean_absolute_error(test, sarima_pred))
metrics['RMSE'].append(np.sqrt(mean_squared_error(test, sarima_pred)))
metrics['SMAPE'].append(smape(test, sarima_pred))
metrics['R2'].append(r2_score(test, sarima_pred))
metrics['Theil_U'].append(theil_u(test, sarima_pred))

prophet_df = pd.DataFrame({'ds': train.index, 'y': train.values})
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=len(test), freq='M')
prophet_pred = prophet_model.predict(future)
prophet_pred = prophet_pred.tail(len(test))['yhat']
prophet_pred.index = test.index
metrics['Model'].append('Prophet')
metrics['MAE'].append(mean_absolute_error(test, prophet_pred))
metrics['RMSE'].append(np.sqrt(mean_squared_error(test, prophet_pred)))
metrics['SMAPE'].append(smape(test, prophet_pred))
metrics['R2'].append(r2_score(test, prophet_pred))
metrics['Theil_U'].append(theil_u(test, prophet_pred))
ma = ts.rolling(window=12, center=False).mean()
X = np.arange(len(train)).reshape(-1, 1)
y = train.values
lr = LinearRegression()
lr.fit(X, y)
X_test = np.arange(len(train), len(ts)).reshape(-1, 1)
lr_pred = lr.predict(X_test)
lr_pred = pd.Series(lr_pred, index=test.index)

metrics['Model'].append('Linear Regression')
metrics['MAE'].append(mean_absolute_error(test, lr_pred))
metrics['RMSE'].append(np.sqrt(mean_squared_error(test, lr_pred)))
metrics['SMAPE'].append(smape(test, lr_pred))
metrics['R2'].append(r2_score(test, lr_pred))
metrics['Theil_U'].append(theil_u(test, lr_pred))

plt.figure(figsize=(12, 6))
plt.plot(train, label='Обучающая выборка')
plt.plot(test, label='Тестовая выборка')
plt.plot(arima_pred, label='ARIMA')
plt.plot(sarima_pred, label='SARIMA')
plt.plot(prophet_pred, label='Prophet')
plt.plot(lr_pred, label='Линейная регрессия')
plt.title('Прогнозы моделей')
plt.xlabel('Год')
plt.ylabel('Пассажиры (тыс.)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(ts, label='Исходный ряд')
plt.plot(ma, label='Скользящее среднее')
plt.title('Скользящее среднее')
plt.xlabel('Год')
plt.ylabel('Пассажиры (тыс.)')
plt.legend()
plt.show()

print("\nМетрики моделей:")
metrics_df = pd.DataFrame(metrics)
print(metrics_df)

