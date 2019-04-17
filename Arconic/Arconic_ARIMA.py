
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from matplotlib import pyplot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA

#Training data load
df_arima = pd.read_csv('C:/yoyo/course/615/project/arconic/data/Arconic.csv')
alist=[12300,12600,11286,14388,12610,12310,12354,11280,14403,14386,14407,11281,12630,12353,11285,14405]
df_arima = df_arima[df_arima['Item Group (Product Family)'].isin(alist)]
df_arima = df_arima[['Item Group (Product Family)','Quantity (PCS)','Request Date','Year','Week','Month','Quarter']]
df_arima.rename(columns={'Item Group (Product Family)': 'Item','Quantity (PCS)': 'Quantity',
                  'Request Date': 'Date'}, inplace=True)
df_arima = df_arima[(df_arima['Year'] >= 2015) & (df_arima['Year'] <= 2017)]
df_arima.head()


#Test data load
df_pre = pd.read_csv('C:/yoyo/course/615/project/arconic/data/Arconic.csv')
alist=[12300,12600,11286,14388,12610,12310,12354,11280,14403,14386,14407,11281,12630,12353,11285,14405]
df_pre = df_pre[df_pre['Item Group (Product Family)'].isin(alist)]
df_pre = df_pre[['Item Group (Product Family)','Quantity (PCS)','Request Date','Year','Week','Month','Quarter']]
df_pre.rename(columns={'Item Group (Product Family)': 'Item','Quantity (PCS)': 'Quantity',
                  'Request Date': 'Date'}, inplace=True)

#ARIMA Model Function define
def arima(df,item,p):
    train_week=df.loc[item]
    train_week=pd.pivot_table(train_week, index='Date', values='Quantity')
    model = ARIMA(train_week, order=(p,1,0))
    model_fit = model.fit(disp=0)
    return model_fit

##MAPE define
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

##weekly ARIMA
df_pre_week = df_pre[(df_pre['Year'] == 2018) & (df_pre['Week'] <= 6)]
df_pre_week=pd.pivot_table(df_pre_week, values=['Quantity', 'Date'], index=['Item', 'Year','Week'],
                 aggfunc={'Quantity': sum,                       
                          'Date': max})
df_pre_week['Date'] = df_pre_week['Date'].apply(lambda x: pd.to_datetime(x))
df_test_week = df_pre_week
df_train_week =pd.pivot_table(df_arima, values=['Quantity', 'Date'], index=['Item', 'Year','Week'],
                 aggfunc={'Quantity': sum,                       
                          'Date': max})
df_train_week['Date'] = df_train_week['Date'].apply(lambda x: pd.to_datetime(x))
print('--Weekly predict MAPE by first 6 weeks in 2018--')
print('item',' MAPE',' Forecast')
mape = 0
for item in alist:
    model_fit = arima(df_train_week,item,12)
    test_week=df_test_week.loc[item]
    test = pd.pivot_table(test_week,index = 'Date',values ='Quantity')
    y_test = test['Quantity']
    y_pred = model_fit.forecast(len(y_test))[0]
    item_MAPE = round(mean_absolute_percentage_error(y_test, y_pred),2)
    mape = mape + item_MAPE
    print(item,item_MAPE)

##Monthly ARIMA
df_pre_month = df_pre[(df_pre['Year'] == 2018) & (df_pre['Month'] <= 6)]
df_pre_month=pd.pivot_table(df_pre_month, values=['Quantity', 'Date'], index=['Item', 'Year','Month'],
                 aggfunc={'Quantity': sum,                       
                          'Date': max})
df_pre_month['Date'] = df_pre_month['Date'].apply(lambda x: pd.to_datetime(x))
df_test_month = df_pre_month
df_train_month =pd.pivot_table(df_arima, values=['Quantity', 'Date'], index=['Item', 'Year','Month'],
                 aggfunc={'Quantity': sum,                       
                          'Date': max})
df_train_month['Date'] = df_train_month['Date'].apply(lambda x: pd.to_datetime(x))
print('--Monthly predict MAPE by first 6 months in 2018--')
print('item',' MAPE')
mape = 0
for item in alist:
    model_fit = arima(df_train_month,item,6)
    test_month=df_test_month.loc[item]
    test = pd.pivot_table(test_month,index = 'Date',values ='Quantity')
    y_test = test['Quantity']
    y_pred = model_fit.forecast(len(y_test))[0]
    item_MAPE = round(mean_absolute_percentage_error(y_test, y_pred),2)
    mape = mape + item_MAPE
    print(item_MAPE)

##Quarterly ARIMA
df_pre_qr = df_pre[(df_pre['Year'] == 2018) & (df_pre['Quarter'] <= 2)]
df_pre_qr=pd.pivot_table(df_pre_qr, values=['Quantity', 'Date'], index=['Item', 'Year','Quarter'],
                 aggfunc={'Quantity': sum,                       
                          'Date': max})
df_pre_qr['Date'] = df_pre_qr['Date'].apply(lambda x: pd.to_datetime(x))
df_test_qr = df_pre_qr
df_train_qr =pd.pivot_table(df_arima, values=['Quantity', 'Date'], index=['Item', 'Year','Quarter'],
                 aggfunc={'Quantity': sum,                       
                          'Date': max})
df_train_qr['Date'] = df_train_qr['Date'].apply(lambda x: pd.to_datetime(x))
print('--Quaterly predict MAPE by first 2 quarters in 2018--')
print('item',' MAPE')
mape = 0
for item in alist:
    model_fit = arima(df_train_qr,item,2)
    test_qr=df_test_qr.loc[item]
    test = pd.pivot_table(test_qr,index = 'Date',values ='Quantity')
    y_test = test['Quantity']
    y_pred = model_fit.forecast(len(y_test))[0]
    item_MAPE = round(mean_absolute_percentage_error(y_test, y_pred),2)
    mape = mape + item_MAPE
    print(item,item_MAPE)

