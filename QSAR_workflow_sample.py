"""
QSAR Workflow Sample (Fixed)
Author: Divya Karade (adapted)
Description: Cleaned version of the original script with bugs fixed.
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
"""

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from numpy.random import seed
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import tensorflow as tf
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn import metrics
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys

from keras.layers import Dense

# Disable RDKit logging to keep output clean
RDLogger.DisableLog("rdApp.*")


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


class RDKit_2D:
    def __init__(self, smiles):
        self.mols = [Chem.MolFromSmiles(i) for i in smiles]
        self.smiles = smiles

    def compute_2Drdkit(self, name):
        rdkit_2d_desc = []
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        header = calc.GetDescriptorNames()
        for i in range(len(self.mols)):
            ds = calc.CalcDescriptors(self.mols[i])
            rdkit_2d_desc.append(ds)
        df = pd.DataFrame(rdkit_2d_desc, columns=header)
        df.insert(loc=0, column='smiles', value=self.smiles)
        return df

    def compute_MACCS(self, name):
        MACCS_list = []
        header = ['bit' + str(i) for i in range(167)]
        for i in range(len(self.mols)):
            ds = list(MACCSkeys.GenMACCSKeys(self.mols[i]).ToBitString())
            MACCS_list.append(ds)
        df2 = pd.DataFrame(MACCS_list, columns=header)
        df2.insert(loc=0, column='smiles', value=self.smiles)
        return df2


# -------------------------
# Reproducibility: seeds & TF session (kept as in original)
# -------------------------
seed(0)
tf.random.set_seed(1)
np.random.seed(3)

# -------------------------
# Load model dataset & build model descriptors for model training data
# -------------------------
data = pd.read_csv("SpikeRBD_DD.csv")

# Compute RDKit descriptors for the model dataset (training data)
# Assumes RDKit_2D accepts a list/series of smiles for initialization
RDKit_descriptor = RDKit_2D(data['smiles'])
x1 = RDKit_descriptor.compute_2Drdkit(data)
x2 = RDKit_descriptor.compute_MACCS(data)
x3 = x2.iloc[:, 1:]
x4 = pd.concat([data['DockingScore'], x1, x3], axis=1)

print('2-D descriptors and MACCS fingerprints generated for model data:')
print(x4)

# -------------------------
# Prepare features (X) and labels (Y) for model training
# -------------------------
labels = x4['DockingScore']
features = x4.iloc[:, 3:]  # keep as original
# ensure numeric
features = features.apply(pd.to_numeric, errors='coerce')
print(f"Features shape before cleaning: {features.shape}")

# Clean features: remove inf/NaN, clip extremes
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.dropna(inplace=True)
features = features.clip(lower=-1e10, upper=1e10)
X = features.astype(np.float64)

# Align labels with cleaned features
labels = labels.loc[features.index]
Y = np.ravel(labels).astype(np.float64)
Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

# -------------------------
# INPUT SMILES (User or CSV)
# -------------------------

smiles = ["CCO"]  # example ethanol SMILES

valid_smiles = []
invalid_rows = []

for i, s in enumerate(smiles):
    if pd.isna(s) or str(s).strip() == "":
        invalid_rows.append((i, s))
        continue
    mol = Chem.MolFromSmiles(str(s).strip())
    if mol is None:
        invalid_rows.append((i, s))
    else:
        valid_smiles.append(str(s).strip())

if invalid_rows:
    print("⚠️ Invalid or unparsable SMILES removed:")
    for idx, s in invalid_rows:
        print(f"Row {idx}: `{s}`")

if len(valid_smiles) == 0:
    print("No valid SMILES after cleaning. Aborting descriptor generation.")
    print()

# Rebuild cleaned df3 used by descriptor functions (ensures format expected by RDKit_2D)
df3 = pd.DataFrame({"smiles": valid_smiles})
smiles = df3['smiles'].values  # update variable used later

# Save cleaned .smi (tab-separated) for downstream tools (as before)
df3.to_csv('molecule.smi', sep='\t', header=False, index=False)

# -------------------------
# Compute descriptors for input SMILES — with safe try/except
# -------------------------
try:
    RDKit_descriptor = RDKit_2D(smiles)
    x5 = RDKit_descriptor.compute_2Drdkit(df3)
    x6 = RDKit_descriptor.compute_MACCS(df3)
except Exception as e:
    print(
        "Error while computing RDKit descriptors. One or more SMILES may be invalid or RDKit raised an error.")
    print(e)
    print()

x7 = x6.iloc[:, 1:]
x8 = pd.concat([x5, x7], axis=1)

print('2-D descriptors and MACCS fingerprints generated for input SMILES:')
print(x8)

# -------------------------
# Train/test split and scaling (same logic as original, but cleaned)
# -------------------------

# Split the data with fixed random state
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Ensure that there are no inf or NaN values in the scaled data
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Identify outliers in the training dataset
iso = IsolationForest(contamination=0.1, n_estimators=100, random_state=3, verbose=0)
pca = PCA(n_components=2)
pca1 = pca.fit_transform(X_train)
yhat_train = iso.fit_predict(pca1)
pca2 = pca.fit_transform(X_test)
yhat_test = iso.fit_predict(pca2)

# Select all rows that are not outliers
mask_train = yhat_train != -1
X_train, y_train = X_train[mask_train, :], y_train[mask_train]

mask_test = yhat_test != -1
X_test, y_test = X_test[mask_test, :], y_test[mask_test]

# Summarize the shape of the updated training & test dataset
print('Updated training & test dataset after removal of outliers')
print("Training set after outlier removal: {}".format(X_train.shape))
print("Test set after outlier removal: {}".format(X_test.shape))

# Convert x4 and data columns to numeric, invalid parsing will be set as NaN
x10 = x8.apply(pd.to_numeric, errors='coerce')
x4 = x4.apply(pd.to_numeric, errors='coerce')

# Extract the columns from 'MaxAbsEStateIndex' onwards
descriptor_columns = x10.loc[:, 'MaxAbsEStateIndex':]
model_data_columns = x4.loc[:, 'MaxAbsEStateIndex':]

# Initialize a flag to check if all values are within the range
all_in = True

# Iterate through each column in descriptor_columns
for column in descriptor_columns.columns:
    if column in model_data_columns.columns:
        min_value = model_data_columns[column].min()
        max_value = model_data_columns[column].max()
        if not ((descriptor_columns[column] >= min_value) & (
                descriptor_columns[column] <= max_value)).all():
            all_in = False
            break

# Determine the result based on the all_in flag
applicability_domain_result = 'IN' if all_in else 'OUT'

# Create the final DataFrame with input SMILES and applicability domain result
final_df = pd.DataFrame({
    'Input SMILES': smiles,
    'Applicability Domain': applicability_domain_result
})

print('Applicability domain information for input SMILES')
# Display results in the Streamlit app
print("**Applicability domain (PCA & Isolation forest)**")
print(final_df)

# Define the model
model = Sequential()
model.add(Dense(600, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mape', rmse, r_square])

# Enable early stopping based on r_square
earlystopping = EarlyStopping(monitor='val_r_square', patience=200, verbose=1, mode='max')

# Train the model
result = model.fit(X_train, y_train, epochs=200, batch_size=400, shuffle=True, verbose=2,
                   validation_data=(X_test, y_test), callbacks=[earlystopping])

# Predict on new data
x9 = sc.transform(x8.iloc[:, 2:])
ACEpredict = model.predict(x9)
deepdock = pd.DataFrame(ACEpredict, columns=['DeepDock Prediction'])
deepdock.insert(loc=0, column='smiles', value=smiles)
# Set a subheader and display the regression
print('Prediction:\nBinding affinity of input data against selected protein target\n ')
print("**DeepDock Prediction**")
print(deepdock)
print('Prediction Created Successfully!')

# Make predictions with the neural network
y_pred = model.predict(X_test)
x_pred = model.predict(X_train)

print("**Training set model validated based on both training and test set**")

# Model summary
s = io.StringIO()
model.summary(print_fn=lambda x: s.write(x + '\n'))
model_summary = s.getvalue()
s.close()

print('Model summary')
print(model_summary)

print('Model prediction evaluation:')

# Print statistical figures of merit for training set
print('Trained_error_rate:')
print("Mean absolute error (MAE): %f" % sklearn.metrics.mean_absolute_error(y_train, x_pred))
print("Mean squared error (MSE): %f" % sklearn.metrics.mean_squared_error(y_train, x_pred))
print("Root mean squared error (RMSE): %f" % math.sqrt(
    sklearn.metrics.mean_squared_error(y_train, x_pred)))
print("Coefficient of determination ($R^2$): %f" % sklearn.metrics.r2_score(y_train, x_pred))

# Print statistical figures of merit for test set
print('Test_error_rate:')
print("Mean absolute error (MAE): %f" % sklearn.metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error (MSE): %f" % sklearn.metrics.mean_squared_error(y_test, y_pred))
print("Root mean squared error (RMSE): %f" % math.sqrt(
    sklearn.metrics.mean_squared_error(y_test, y_pred)))
print("Coefficient of determination ($R^2$): %f" % sklearn.metrics.r2_score(y_test, y_pred))

# Print final results
print("Done.")

