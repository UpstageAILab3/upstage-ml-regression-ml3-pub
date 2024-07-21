# **🏠 부동산 실거래가 Team 3 DTQ First Trial code**

# ## Contents

# ## Step 1: Library Imports
# - 필요한 라이브러리를 불러옵니다.

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset

import wandb
from datetime import datetime

# Ensure the torch is using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumBarunGothic')                        # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe)              # Matplotlib에폰트 추가
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'}) # 폰트 설정
plt.rc('font', family='NanumBarunGothic')
import seaborn as sns

# utils
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import warnings; warnings.filterwarnings('ignore')

# Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import eli5
from eli5.sklearn import PermutationImportance

# Hyperparameters
batch_size = 32
learning_rate = 0.015
num_epochs = 3
hidden_units = [64]
dropout_type = 'normal'
hidden_activation = 'prelu'
early_stopping = 3
max_batch_size = 131072
optimizer_type = 'adam'
loss_function = 'mean_squared_error'

# Initialize wandb
wandb.init(project="real-estate-prices", config={
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "hidden_units": hidden_units,
    "dropout_type": dropout_type,
    "hidden_activation": hidden_activation,
    "early_stopping": early_stopping,
    "max_batch_size": max_batch_size,
    "optimizer_type": optimizer_type,
    "loss_function": loss_function
})

# ## Step 2: Data Preprocessing Functions

# Optimize data types
def optimize_dtypes(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int32)
    return df

def one_hot_encode(df, columns):
    for col in columns:
        encoded_df = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
    return df

def impute_missing_values(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
            print(f"{datetime.now()} - Imputed missing values in {column} with median value {median_value}")
    return df

def smooth_ridit_transform(df, columns):
    for col in columns:
        sorted_col = df[col].sort_values()
        rank = sorted_col.rank(pct=True)
        ridit_score = 2 * rank - 1
        df[col] = ridit_score
    return df

def bin_numerical_variables(df, columns, n_bins=5):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    for col in columns:
        df[col + '_bin'] = discretizer.fit_transform(df[[col]]).astype(int)
    return df

def tfidf_transform(df, text_column):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df

# ## Step 3: Custom Dataset Class for PyTorch

class RealEstateDataset(Dataset):
    def __init__(self, dataframe, target_column=None):
        self.dataframe = dataframe
        self.target_column = target_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataframe.iloc[idx]
        if self.target_column:
            features = sample.drop(self.target_column).values
            target = sample[self.target_column]
            return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
        else:
            features = sample.values
            return torch.tensor(features, dtype=torch.float32)

# ## Step 4: Define the PyTorch Model

class SimpleResidualNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SimpleResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units[0])
        self.hidden_activation = nn.PReLU() if hidden_activation == 'prelu' else nn.ReLU()
        self.fc2 = nn.Linear(hidden_units[0], 1)
        self.residual = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        out = self.hidden_activation(self.fc1(x))
        out = self.fc2(out) + self.residual(x)
        return out

# ## Step 5: Training and Inference Functions


def train_model(model, dataloader, criterion, optimizer, num_epochs=25, log_interval=10):
    model.train()
    rmse_history = []
    best_rmse = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_rmse = torch.sqrt(loss).item()
            epoch_losses.append(batch_rmse)

            # 주기적 로깅
            if batch_idx % log_interval == 0:
                wandb.log({"epoch": epoch, "batch_idx": batch_idx, "batch_rmse": batch_rmse})

        epoch_rmse = np.mean(epoch_losses)
        rmse_history.append(epoch_rmse)
        wandb.log({"epoch": epoch+1, "epoch_rmse": epoch_rmse})
        print(f"{datetime.now()} - Epoch [{epoch+1}/{num_epochs}], RMSE: {epoch_rmse:.4f}")

        if epoch_rmse < best_rmse:
            best_rmse = epoch_rmse
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if early_stopping > 0 and epochs_no_improve >= early_stopping:
            print(f"{datetime.now()} - Early stopping at epoch {epoch+1}")
            break

    return model, rmse_history

def plot_rmse(rmse_history):
    plt.figure(figsize=(10, 5))
    plt.plot(rmse_history, label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE over Epochs')
    plt.legend()
    plt.show()

def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
    return np.array(predictions)

# ## Step 6: Putting It All Together

# Load data in chunks
train_data = pd.read_csv('/data/ephemeral/home/train.csv')
test_data = pd.read_csv('/data/ephemeral/home/test.csv')

# Select only the required columns
selected_columns = ['시군구', '번지', '아파트명', '전용면적(㎡)', '계약년월', '건축년도', 'target']
train_data = train_data[selected_columns]
test_data = test_data[['시군구', '번지', '아파트명', '전용면적(㎡)', '계약년월', '건축년도']]

train_data = optimize_dtypes(train_data)
test_data = optimize_dtypes(test_data)

# Log memory usage
print(f"{datetime.now()} - Memory usage after loading data: {train_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

# Data preprocessing
train_data = one_hot_encode(train_data, columns=['시군구', '번지', '아파트명'])

# Log memory usage after one_hot_encode
print(f"{datetime.now()} - Memory usage after one_hot_encode: {train_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

train_data = impute_missing_values(train_data)

# Log memory usage after imputing missing values
print(f"{datetime.now()} - Memory usage after imputing missing values: {train_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

train_data = smooth_ridit_transform(train_data, columns=['전용면적(㎡)', '계약년월', '건축년도'])

# Log memory usage after smooth_ridit_transform values
print(f"{datetime.now()} - Memory usage after smooth_ridit_transform values: {train_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

test_data = one_hot_encode(test_data, columns=['시군구', '번지', '아파트명'])

# Log memory usage after one_hot_encode
print(f"{datetime.now()} - Memory usage after one_hot_encode: {test_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

test_data = impute_missing_values(test_data)

# Log memory usage after imputing missing values
print(f"{datetime.now()} - Memory usage after imputing missing values: {test_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

test_data = smooth_ridit_transform(test_data, columns=['전용면적(㎡)', '계약년월', '건축년도'])

# Log memory usage after preprocessing
print(f"{datetime.now()} - Memory usage after preprocessing: {test_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

# Create dataset instances
train_dataset = RealEstateDataset(train_data, target_column='target')
test_dataset = RealEstateDataset(test_data)

print(f"{datetime.now()} - Create dataset instances...Done")

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"{datetime.now()} - Create data loaders...Done")

# Initialize model, criterion, and optimizer
model = SimpleResidualNetwork(input_dim=train_data.shape[1] - 1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
trained_model, rmse_history = train_model(model, train_dataloader, criterion, optimizer, num_epochs=num_epochs)

print(f"{datetime.now()} - Train model...Done")

# Plot RMSE history
plot_rmse(rmse_history)

# Make predictions
predictions = predict(trained_model, test_dataloader)

# Save predictions
np.savetxt('predictions.csv', predictions, delimiter=',')

print(f"{datetime.now()} - Make predictions...Done")
