import pandas as pd
import os
import sys
from sklearn.preprocessing import MinMaxScaler


def get_csv_data(filename, index=0):
    '''
    Gets data from a csv file and puts it into a pandas dataframe
    '''
    if os.path.exists(filename):
        print( filename + " found ")
        data_frame = pd.read_csv(filename, index_col=None)
        return data_frame
    else:
        print("file not found")

def scale_features(df):
    features = list(df.columns[:11])
    scaler = MinMaxScaler()
    X = df[features]
    y = df['quality'].values

    # for i in range(0, len(y)):
    #     if y[i] >= 0 and y[i] <= 5:
    #         y[i] = 0
    #     else:
    #         y[i] = 1

    X_Scale = scaler.fit_transform(X)
    data = pd.DataFrame(X_Scale)
    data.insert(loc=11, column='quality', value=y)
    data.to_csv("winequality_white_scaled.csv", encoding='utf-8', index=False)

df = get_csv_data("winequality_white.csv")
print(df)
scale_features(df)
