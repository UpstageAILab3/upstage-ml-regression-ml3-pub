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
import math
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
    vectorizer = TfidfVectorizer(max_features=20000)
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df

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

class SimpleResidualNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SimpleResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units[0])
        self.hidden_activation = nn.PReLU() if hidden_activation == 'prelu' else nn.ReLU()
        self.fc2 = nn.Linear(hidden_units[0], 1)
        self.residual = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.hidden_activation(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out) + self.residual(x)
        return out

def train_model(model, dataloader, criterion, optimizer, num_epochs=25, log_interval=10):
    model.train()
    rmse_history = []
    batch_rmse_history = []
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
            batch_rmse_history.append(batch_rmse)

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

    return model, rmse_history, batch_rmse_history

def plot_rmse(rmse_history, batch_rmse_history):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rmse_history, label='Epoch RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Epoch RMSE over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(batch_rmse_history, label='Batch RMSE', color='orange')
    plt.xlabel('Batch')
    plt.ylabel('RMSE')
    plt.title('Batch RMSE over Batches')
    plt.legend()

    plt.tight_layout()
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

selected_features = ['시군구', '번지', '아파트명', '전용면적(㎡)', '계약년월', '건축년도', '좌표X', '좌표Y']
lead_house = {
    "강서구": (37.56520754904415, 126.82349451366355),
    "관악구": (37.47800896704934, 126.94178722423047),
    "강남구": (37.530594054209146, 127.0262701317293),
    "강동구": (37.557175745977375, 127.16359581113558),
    "광진구": (37.543083184171, 127.0998363490422),
    "구로구": (37.51045944660659, 126.88687199829572),
    "금천구": (37.459818907487936, 126.89741481874103),
    "노원구": (37.63952738902813, 127.07234254197617),
    "도봉구": (37.65775043994647, 127.04345013224447),
    "동대문구": (37.57760781415707, 127.05375628992316),
    "동작구": (37.509881249641495, 126.9618159122961),
    "마포구": (37.54341664563958, 126.93601641235335),
    "서대문구": (37.55808950436837, 126.9559315685538),
    "서초구": (37.50625410912666, 126.99846468032919),
    "성동구": (37.53870643389788, 127.04496220606433),
    "성북구": (37.61158435092128, 127.02699796439015),
    "송파구": (37.512817775046074, 127.08340371063358),
    "양천구": (37.526754982736556, 126.86618704123521),
    "영등포구": (37.52071403351804, 126.93668907644046),
    "용산구": (37.521223570097305, 126.97345317787784),
    "은평구": (37.60181702377437, 126.9362806808709),
    "종로구": (37.56856915384472, 126.96687674967252),
    "중구": (37.5544678205846, 126.9634879236162),
    "중랑구": (37.58171824083332, 127.08183326205129),
    "강북구": (37.61186335979484, 127.02822407466175)
}

def haversine_distance(lat1, lon1, lat2, lon2):
    radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = radius * c
    return distance

def preprocess_data(df, include_target=False):
    if include_target:
        df = df[selected_features + ['target']]
    else:
        df = df[selected_features]
    df[["시", "구", "동"]] = df["시군구"].str.split(" ", expand=True)
    lead_house_data = pd.DataFrame([{"구": k, "대장_좌표X": v[1], "대장_좌표Y": v[0]} for k, v in lead_house.items()])
    df = pd.merge(df, lead_house_data, how="left", on="구")
    df['대장아파트_거리'] = df.apply(lambda row: haversine_distance(row["좌표Y"], row["좌표X"], row["대장_좌표Y"], row["대장_좌표X"]) if pd.notnull(row["좌표Y"]) and pd.notnull(row["좌표X"]) else np.nan, axis=1)
    df['좌표X'].fillna(df['대장_좌표X'], inplace=True)
    df['좌표Y'].fillna(df['대장_좌표Y'], inplace=True)
    df = optimize_dtypes(df)
    df = one_hot_encode(df, columns=['시군구', '번지', '아파트명'])
    df = impute_missing_values(df)
    df = smooth_ridit_transform(df, columns=['전용면적(㎡)', '계약년월', '건축년도'])
    return df

def remove_outliers(df, z_thresh=3):
    mask = (df['계약년월일'] >= '2021-01-01') & (df['계약년월일'] <= '2022-01-31')
    z_scores = np.abs(stats.zscore(df.loc[mask, 'target']))
    df = df[~((z_scores >= z_thresh) & mask)]
    return df

train_data = pd.read_csv('/data/ephemeral/home/train.csv')
test_data = pd.read_csv('/data/ephemeral/home/test.csv')

train_data = preprocess_data(train_data, include_target=True)
test_data = preprocess_data(test_data, include_target=False)

# Check for '계약일' column and handle its absence
if '계약일' in train_data.columns:
    train_data['계약일'] = train_data['계약일'].apply(lambda x: f'{x:02d}')
    train_data["계약년월일"] = train_data["계약년월"].astype(str) + train_data["계약일"].astype(str)
    train_data['계약년월일'] = pd.to_datetime(train_data['계약년월일'])
else:
    train_data["계약연도"] = train_data["계약년월"].astype(str).str[:4]
    train_data["계약월"] = train_data["계약년월"].astype(str).str[4:]
    train_data["계약년월일"] = train_data["계약년월"].astype(str) + '01'  # Default to the first day of the month
    train_data['계약년월일'] = pd.to_datetime(train_data['계약년월일'])

if '계약일' in test_data.columns:
    test_data['계약일'] = test_data['계약일'].apply(lambda x: f'{x:02d}')
    test_data["계약년월일"] = test_data["계약년월"].astype(str) + test_data["계약일"].astype(str)
    test_data['계약년월일'] = pd.to_datetime(test_data['계약년월일'])
else:
    test_data["계약연도"] = test_data["계약년월"].astype(str).str[:4]
    test_data["계약월"] = test_data["계약년월"].astype(str).str[4:]
    test_data["계약년월일"] = test_data["계약년월"].astype(str) + '01'  # Default to the first day of the month
    test_data['계약년월일'] = pd.to_datetime(test_data['계약년월일'])

train_data = remove_outliers(train_data)

train_dataset = RealEstateDataset(train_data, target_column='target')
test_dataset = RealEstateDataset(test_data)

print(f"{datetime.now()} - Create dataset instances...Done")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"{datetime.now()} - Create data loaders...Done")

model = SimpleResidualNetwork(input_dim=train_data.shape[1] - 1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(next(model.parameters()).is_cuda)

trained_model, rmse_history, batch_rmse_history = train_model(model, train_dataloader, criterion, optimizer, num_epochs=num_epochs)

print(f"{datetime.now()} - Train model...Done")

plot_rmse(rmse_history, batch_rmse_history)

predictions = predict(trained_model, test_dataloader)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
file_name = f'predictions_{timestamp}.csv'
predictions_df = pd.DataFrame(predictions, columns=['target'])
predictions_df.to_csv(file_name, index=False)

print(f"{datetime.now()} - Saved predictions to {file_name}...Done")
