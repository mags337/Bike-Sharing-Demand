import numpy as np
import math

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, KBinsDiscretizer

#Pipeline
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures

#Transform
from sklearn.compose import make_column_transformer, ColumnTransformer


def add_time_features(df):
    '''
    Extracts features like hour, month etc. from the datetime column.
    '''
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['day_of_week'] = df.index.day_name()
    df['hour'] = df.index.hour

    df['hour_cos'] = np.cos(2 * math.pi * df['hour'] / df['hour'].max())
    df['hour_sin'] = np.sin(2 * math.pi * df['hour']/24)

    df['month_sin'] = np.sin(2 * math.pi  * df.index.month/12)
    df['month_cos'] = np.cos(2 * math.pi  * df.index.month/12)

    return df



def feature_engineering(df):
    '''
    sss
    '''
    # Defining transformers and features
    num_tr = MinMaxScaler()
    num_fe = df["temp", "atemp", "humidity", "windspeed"]

    time_tr = KBinsDiscretizer(n_bins=12, encode='onehot')
    time_fe = df["hour", "month"]

    #Creating column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tr, num_fe),
            ("time", time_tr, time_fe)
            ], 
            remainder="passthrough")
        
    return 


def engineer_df(df):
    '''
    sss
    '''

    df = df.add_time_features(df)