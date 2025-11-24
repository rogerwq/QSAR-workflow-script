"""
QSAR Workflow Sample
Author: Divya Karade
Description:
A clean, minimal QSAR pipeline demonstrating:
- Data loading
- Descriptor generation (RDKit 2D + MACCS)
- Feature cleaning
- Train/test split
- Scaling
- Outlier removal (PCA + Isolation Forest)
- Applicability Domain (AD)
- Neural network model training (Keras)
- Evaluation
- Prediction for new SMILES

This sample is for workflow structuring only.
"""

import numpy as np
import pandas as pd
import math
from rdkit import Chem
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import sklearn.metrics

from numpy.random import seed

# -------------------------
# Custom metrics
# -------------------------
def rmse(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def r_square(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


# -------------------------
# Reproducibility
# -------------------------
seed(0)
tf.random.set_seed(1)
np.random.seed(3)


# -------------------------
# Load dataset
# -------------------------
def load_dataset(path):
    data = pd.read_csv(path)
    return data


# -------------------------
# Descriptor generation wrapper
# -------------------------
class DescriptorGenerator:
    def __init__(self, smiles_series):
        self.smiles = smiles_series

    def compute_rdkit_2d(self):
        from rdkit.ML.Descriptors import MoleculeDescriptors
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        descriptor_names = [desc[0] for desc in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

        desc_values = []
        for sm in self.smiles:
            mol = Chem.MolFromSmiles(sm)
            if mol:
                desc_values.append(calculator.CalcDescriptors(mol))
            else:
                desc_values.append([np.nan] * len(descriptor_names))

        df = pd.DataFrame(desc_values, columns=descriptor_names)
        return df

    def compute_maccs(self):
        from rdkit.Chem import MACCSkeys
        fps = []
        for sm in self.smiles:
            mol = Chem.MolFromSmiles(sm)
            if mol:
                bitvect = MACCSkeys.GenMACCSKeys(mol)
                fps.append(list(bitvect))
            else:
                fps.append([np.nan] * 167)

        cols = ["MACCS_" + str(i) for i in range(167)]
        return pd.DataFrame(fps, columns=cols)


# -------------------------
# Applicability Domain
# -------------------------
def check_applicability_domain(model_df, input_df):
    model_cols = model_df.loc[:, 'MaxAbsEStateIndex':]
    input_cols = input_df.loc[:, 'MaxAbsEStateIndex':]

    in_ad = True
    for c in input_cols.columns:
        if c in model_cols.columns:
            if not ((input_cols[c] >= model_cols[c].min()) &
                    (input_cols[c] <= model_cols[c].max())).all():
                in_ad = False
                break
    return "IN" if in_ad else "OUT"


# -------------------------
# Model Training
# -------------------------
def train_qsar_model(X_train, y_train, X_test, y_test):

    model = Sequential([
        Dense(600, activation='relu', input_dim=X_train.shape[1]),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mae', 'mape', rmse, r_square]
    )

    earlystop = EarlyStopping(
        monitor='val_r_square',
        patience=50,
        mode='max',
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=256,
        validation_data=(X_test, y_test),
        callbacks=[earlystop],
        verbose=1
    )

    return model, history


# -------------------------
# Main QSAR Pipeline
# -------------------------
def run_qsar_workflow(dataset_path, input_smiles_list):

    # Load dataset
    df = load_dataset(dataset_path)

    # ---- Descriptors for training data ----
    desc = DescriptorGenerator(df['smiles'])
    rdkit2d = desc.compute_rdkit_2d()
    maccs = desc.compute_maccs().iloc[:, 1:]
    train_df = pd.concat([df['DockingScore'], rdkit2d, maccs], axis=1)

    labels = train_df['DockingScore']
    features = train_df.iloc[:, 3:].apply(pd.to_numeric, errors='coerce')

    # Clean
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)
    labels = labels.loc[features.index]

    X = features.values.astype(float)
    Y = labels.values.astype(float)

    # Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Outliers
    iso = IsolationForest(contamination=0.1, random_state=3)
    pca = PCA(n_components=2)

    yhat_train = iso.fit_predict(pca.fit_transform(X_train))
    mask_train = yhat_train != -1

    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    # ---- Input SMILES Descriptors ----
    valid_smiles = []
    for s in input_smiles_list:
        if Chem.MolFromSmiles(s):
            valid_smiles.append(s)

    input_df = pd.DataFrame({"smiles": valid_smiles})

    input_desc = DescriptorGenerator(input_df['smiles'])
    in_rdkit = input_desc.compute_rdkit_2d()
    in_maccs = input_desc.compute_maccs().iloc[:, 1:]
    input_full = pd.concat([in_rdkit, in_maccs], axis=1)

    # Applicability Domain
    ad_result = check_applicability_domain(train_df, input_full)

    # Train model
    model, history = train_qsar_model(X_train, y_train, X_test, y_test)

    # Predict input
    X_in = scaler.transform(input_full.iloc[:, 2:])
    predictions = model.predict(X_in)

    return predictions, ad_result, model, history


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    preds, ad, model, hist = run_qsar_workflow(
        dataset_path="SpikeRBD_DD.csv",
        input_smiles_list=[
            "CCO",
            "CCN(CC)CC",
            "Nc1ccc(Cl)cc1"
        ]
    )

    print("\nPredictions:", preds.flatten())
    print("Applicability Domain:", ad)