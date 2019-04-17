import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math

df = pd.read_excel('project1.xlsx')
item_group = [12300, 12600, 11286, 14388, 12610, 12310, 12354, 11280,
              14403, 14386, 14407, 11281, 12630, 12353, 11285, 14405]

#
df_s = df.loc[:,['Item Group (Product Family)','Request Date', 'Quantity (PCS)', 'Month', 'Year', 'Quarter']]
df_1= df_s.loc[(df_s['Year'] > 2014) & (df_s['Year'] < 2019)]
df_2 = df_1.reset_index(drop = True)
l1 = df_2.shape[0]

for j in range(len(item_group)):
    for i in range(12):
        df_2.loc[l1 +i + j *12] = {'Item Group (Product Family)': item_group[j], 'Request Date':0,
                         'Quantity (PCS)':0, 'Month': i+1 ,'Year': 2019 , 'Quarter': (int(i/3) +1)}
l2 = df_2.shape[0]
for j in range(len(item_group)):
    for i in range(12):
        df_2.loc[l2 +i + j *12] = {'Item Group (Product Family)': item_group[j], 'Request Date':0,
                         'Quantity (PCS)':0, 'Month': i+1 ,'Year': 2020 , 'Quarter': (int(i/3) +1)}

df_top = df_2.set_index(['Item Group (Product Family)'])

# Define the plot function
def plot_point(x, y, y_pred):
    plt.figure()
    plt.scatter(x, y, color='black')
    plt.plot(x, y_pred, color='blue', linewidth=3)
    plt.xlabel('Time')
    plt.ylabel('Quantity')
    #设置坐标轴刻度
    plt.show()

# Regression Model with Seasonal Factor
def regression_with_seasonal_factor(df, df_new, item, time_unit):
    df_item = df.loc[item]
    df_item_group = df_item.groupby(['Year', time_unit]).sum()

    # Data Preprocessing
    nominal_data = [[str(df_item_group.index[i][1])] for i in range(df_item_group.shape[0])]
    onehot = preprocessing.OneHotEncoder()
    hot_array = onehot.fit_transform(nominal_data).toarray()

    quantity_data = [[df_item_group['Quantity (PCS)'].values[x]] for x in range(df_item_group.shape[0])]
    data_array = np.hstack((quantity_data, hot_array))
    df_encoded = pd.DataFrame(data_array)
    df_encoded['timeline'] = [x for x in range(1, data_array.shape[0] + 1)]
    tr = len(df_item_group.loc[2015, 'Quantity (PCS)'])

    df_encoded.drop([tr], axis = 1, inplace = True)

    ## Training Dataset
    X_train = df_encoded.iloc[: 3 * tr, 1:]
    y_train = np.ravel(df_encoded.iloc[:3 * tr, 0])

    ## Trainging the model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Validate dataset
    X_validate = df_encoded.iloc[3 * tr:int(3.5 * tr), 1:]
    y_validate = np.ravel(df_encoded.iloc[3 * tr:int(3.5 * tr), 0])

    y_train_pred = lr.pred(X_train)
    y_validate_pred = lr.predict(X_validate)
    # calculate the model parameter
    mse = math.sqrt(mean_squared_error(y_validate, y_validate_pred))
    mape = abs(y_validate - y_validate_pred) / (y_validate)
    r = r2_score(y_train, y_train_pred)

    # Plot the Trend line
    plot_point(X_train.iloc[:,-1], y_train, y_train_pred)
    return item, np.mean(mape), r

def prediction_factor(df, df_new, item, time_unit):
    df_item = df.loc[item]
    df_item_group = df_item.groupby(['Year', time_unit]).sum()

    # Data Preprocessing
    nominal_data = [[str(df_item_group.index[i][1])] for i in range(df_item_group.shape[0])]
    onehot = preprocessing.OneHotEncoder()
    hot_array = onehot.fit_transform(nominal_data).toarray()

    quantity_data = [[df_item_group['Quantity (PCS)'].values[x]] for x in range(df_item_group.shape[0])]
    data_array = np.hstack((quantity_data, hot_array))
    df_encoded = pd.DataFrame(data_array)
    df_encoded['timeline'] = [x for x in range(1, data_array.shape[0] + 1)]
    tr = len(df_item_group.loc[2015, 'Quantity (PCS)'])

    df_encoded.drop([tr], axis=1, inplace=True)
    X = df_encoded.iloc[: 4 * tr, 1:]
    y = np.ravel(df_encoded.iloc[:4 * tr, 0])

    lr = LinearRegression()
    lr.fit(X, y)

    y_pred = lr.predict(X)

    X_test = df_encoded.iloc[4 * tr:, 1:]
    y_test_pred = lr.predict(X_test)

    return item, y_test_pred

# Get the parameter of the model
for item in item_group:
    time_unit_1 = 'Quarter'
    data = regression_with_seasonal_factor(df_top, item, time_unit_1)
    print(data)

# get the prediction results of quarterly quantity in 2019, 2020
for item in item_group:
    time_unit = 'Quarter'
    data = prediction_factor(df_top, item, time_unit)
    print(data)


