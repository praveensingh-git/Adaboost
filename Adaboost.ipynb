import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

def adaboost(df, n_estimators=3):
    
    df = df.copy()
    df['weight'] = 1 / len(df)

    models = []
    alphas = []

    for m in range(n_estimators):

        
        dt = DecisionTreeClassifier(max_depth=1)
        X = df[['X1','X2']]
        y = df['label']
        dt.fit(X, y, sample_weight=df['weight'])

        df['y_pred'] = dt.predict(X)
        error = df.loc[df['label'] != df['y_pred'], 'weight'].sum()
        error = max(error, 1e-10)

        alpha = 0.5 * np.log((1 - error) / error)

        df['weight'] = df.apply(
            lambda row: row['weight'] * np.exp(-alpha)
            if row['label'] == row['y_pred']
            else row['weight'] * np.exp(alpha),
            axis=1
        )

        #Normalize weights
        df['weight'] = df['weight'] / df['weight'].sum()

        models.append(dt)
        alphas.append(alpha)

        #resampling
        df = df.sample(n=len(df), replace=True, weights=df['weight']).reset_index(drop=True)

    return models, alphas
