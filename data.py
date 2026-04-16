import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import kagglehub

def get_data():
    # Download the dataset
    data_file ="SUSY.csv";
    if not(os.path.isfile(data_file)):
        os.system('wget http://archive.ics.uci.edu/static/public/279/susy.zip')
        os.system('unzip susy.zip')
        os.system('gzip -d SUSY.csv.gz')
        os.system('rm susy.zip')

    # Load dataframe
    cols = ['targets','lepton  1 pT', 'lepton  1 eta', 'lepton  1 phi', 'lepton  2 pT',
            'lepton  2 eta', 'lepton  2 phi', 'missing energy magnitude', 'missing energy phi',
            'MET_rel', 'axial MET', 'M_R', 'M_TR_2', 'R', 'MT2', 'S_R',
            'M_Delta_R', 'dPhi_r_b' ,'cos(theta_r1)'];
    data = pd.read_csv(data_file, names = cols)
    return data;

# For demonstration purposes, define the data loading function
def load_and_preprocess_data(df, sample_size=10000, isSelectedColumns = False, n_pca_components=-1):
    # Selected columns
    if isSelectedColumns:
      sel_cols = ['targets', 'lepton  1 pT', 'lepton  2 pT', 'missing energy magnitude',
                  'M_TR_2', 'M_Delta_R',
                  'lepton  1 eta',  'lepton  2 eta'];
      df = df[sel_cols]

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define the target variable
    target = "targets"

    # Sample the dataset
    if sample_size is not None and sample_size < len(df):
      df = df.head(sample_size)
      print(f"Dataset sampled to {sample_size} rows.")

    # Separate features and target
    X = df.drop(columns=[target], errors='ignore')
    y = df[target]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    features = list(X.columns)
    df = pd.DataFrame(data=X_scaled, columns=features);
    df[target] = y
    # Split the data into train and test sets using stratified sampling
    df_train, df_test = train_test_split(df, test_size=0.33, stratify=df[target], random_state=42)
    print(f"Features = {len(features)}")
    return df_train, df_test, features, target


# Define your dataset class
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[self.target].values).float()
        self.X = torch.tensor(dataframe[self.features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]