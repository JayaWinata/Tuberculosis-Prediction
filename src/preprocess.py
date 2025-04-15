from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import yaml
import os

params = yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop('Patient_ID', axis=1)

    categorical_cols = df.select_dtypes(include='object').columns
    cols_to_scale = ['Age', 'Cough_Severity', 'Breathlessness', 'Fatigue', 'Weight_Loss']

    # Create a copy of the original DataFrame for transformations
    df_v1 = df.copy()
    le = LabelEncoder()
    for col in categorical_cols:
        df_v1[col] = le.fit_transform(df_v1[col])

    # Apply one-hot encoding
    def one_hot_encoding(df, categorical_cols):
        for col in categorical_cols:
            dummies = pd.get_dummies(df[col], prefix=col).astype(int)
            col_index = df.columns.get_loc(col) + 1
            for dummy_col in reversed(dummies.columns):
                df.insert(col_index, dummy_col, dummies[dummy_col])
            df.drop(col, axis=1, inplace=True)
        return df

    df_v2 = one_hot_encoding(df.copy(), categorical_cols)

    # Define transformations dynamically
    transformations = [
        (df_v1, None),  # df_v1: Label encoded, no scaling
        (df_v2, None),  # df_v2: One-hot encoded, no scaling
        (df_v1.copy(), MinMaxScaler()),  # df_v3: Label encoded + MinMaxScaler
        (df_v1.copy(), StandardScaler()),  # df_v4: Label encoded + StandardScaler
        (df_v2.copy(), MinMaxScaler()),  # df_v5: One-hot encoded + MinMaxScaler
        (df_v2.copy(), StandardScaler()),  # df_v6: One-hot encoded + StandardScaler
    ]

    # Apply transformations and save to CSV
    os.makedirs(os.path.dirname(output_path[0]), exist_ok=True)
    for i, (df, scaler) in enumerate(transformations):
        if scaler:  # Apply scaling if a scaler is provided
            for col in cols_to_scale:
                df[col] = scaler.fit_transform(df[[col]])
        df.to_csv(output_path[i], index=False)

if __name__ == '__main__':
    preprocess(params['input'], params['output'])