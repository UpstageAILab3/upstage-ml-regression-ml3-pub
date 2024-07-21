# Step 1: Library Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import wandb
import math
from datetime import datetime
from scipy import stats
import os

# Ensure the torch is using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if GPU is available and print the status
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # ttf 파일이 저장되어 있는 경로
    name='NanumBarunGothic')  # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe)  # Matplotlib에폰트 추가
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'})  # 폰트 설정
plt.rc('font', family='NanumBarunGothic')
import seaborn as sns

# Utils
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

print("Step 1: Library Imports - Completed")

# Hyperparameters
batch_size = 32
learning_rate = 0.015
num_epochs = 3
hidden_units = [64, 64]
dropout_type = 'normal'
hidden_activation = 'prelu'
early_stopping = 0
max_batch_size = 131072
optimizer_type = 'adam'
loss_function = 'gamma'

# Step 2: Data Preprocessing Functions
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

project_name = wandb.run.project + "-trail-6th"

# Optimize data types
def optimize_dtypes(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int32)
    return df

def impute_missing_values(df):
    for column in df.columns:
        if df[column].isnull().any():
            if pd.api.types.is_numeric_dtype(df[column]):
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
                print(f"{datetime.now()} - Imputed missing values in {column} with median value {median_value}")
            else:
                mode_value = df[column].mode()[0]
                df[column].fillna(mode_value, inplace=True)
                print(f"{datetime.now()} - Imputed missing values in {column} with mode value {mode_value}")
    return df

def one_hot_encode(df, columns, fitted_encoder=None):
    if fitted_encoder is None:
        encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
        transformed = encoder.fit_transform(df[columns])
    else:
        encoder = fitted_encoder
        transformed = encoder.transform(df[columns])
    
    ohe_df = pd.DataFrame.sparse.from_spmatrix(transformed, columns=encoder.get_feature_names_out(columns))
    df = df.drop(columns, axis=1)
    df = df.reset_index(drop=True)
    ohe_df = ohe_df.reset_index(drop=True)
    df = pd.concat([df, ohe_df], axis=1)
    return df, encoder

def haversine_distance(lat1, lon1, lat2, lon2):
    radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = radius * c
    return distance

def process_contract_date(data):
    data["계약연도"] = data["계약년월"].astype(str).str[:4]  # 첫 4글자는 연도
    data["계약월"] = data["계약년월"].astype(str).str[4:]  # 나머지 글자는 월

    # Check if '계약일' column exists, if not, set default day to '01'
    if '계약일' not in data.columns:
        data['계약일'] = '01'
    else:
        data['계약일'] = data['계약일'].apply(lambda x: '01' if pd.isnull(x) or not (1 <= int(x) <= 31) else f'{int(x):02d}')

    # Create full date column
    data["계약년월일"] = data["계약년월"].astype(str) + data["계약일"].astype(str)
    data['계약년월일'] = pd.to_datetime(data['계약년월일'], errors='coerce')

    return data

def preprocess_data(df, include_target=False):
    # Drop rows with missing values in '번지' and '아파트명'
    df = df.dropna(subset=['번지', '아파트명'])
    
    if include_target:
        df = df[selected_features + ['target', '계약년월일']]
    else:
        df = df[selected_features + ['계약년월일']]
    df[["시", "구", "동"]] = df["시군구"].str.split(" ", expand=True)
    lead_house_data = pd.DataFrame([{"구": k, "대장_좌표X": v[1], "대장_좌표Y": v[0]} for k, v in lead_house.items()])
    df = pd.merge(df, lead_house_data, how="left", on="구")
    df['대장아파트_거리'] = df.apply(lambda row: haversine_distance(row["좌표Y"], row["좌표X"], row["대장_좌표Y"], row["대장_좌표X"]) if pd.notnull(row["좌표Y"]) and pd.notnull(row["좌표X"]) else np.nan, axis=1)
    df['좌표X'].fillna(df['대장_좌표X'], inplace=True)
    df['좌표Y'].fillna(df['대장_좌표Y'], inplace=True)
    df = optimize_dtypes(df)
    df = impute_missing_values(df)

    # '시', '구', '동', '대장아파트_거리','대장_좌표X', '대장_좌표Y' 컬럼 삭제
    df = df.drop(columns=['시', '구', '동', '대장아파트_거리','대장_좌표X', '대장_좌표Y'])

    return df

def remove_outliers(df, z_thresh=3):
    mask = (df['계약년월일'] >= '2021-01-01') & (df['계약년월일'] <= '2022-01-31')
    z_scores = np.abs(stats.zscore(df.loc[mask, 'target']))
    df = df[~((z_scores >= z_thresh) & mask)]
    return df

print("Step 2: Data Preprocessing Functions - Completed")

# Step 3: Custom Dataset Class for PyTorch
class RealEstateDataset(Dataset):
    def __init__(self, dataframe, target_column=None):
        self.dataframe = dataframe
        self.target_column = target_column
        self.dataframe = self.dataframe.apply(pd.to_numeric, errors='coerce')  # Ensure all columns are numeric
        self.dataframe = self.dataframe.fillna(0)  # Fill any NaNs with 0

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

print("Step 3: Custom Dataset Class for PyTorch - Completed")

# Step 4: Define the PyTorch Model
class SimpleResidualNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SimpleResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.residual = nn.Linear(input_dim, hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], 1)
        self.dropout = nn.Dropout(p=0.5)
        self.hidden_activation = nn.PReLU()

    def forward(self, x):
        out = self.hidden_activation(self.fc1(x))
        out = self.hidden_activation(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out) + self.residual(x)
        return out

print("Step 4: Define the PyTorch Model - Completed")

# Step 5: Training and Inference Functions
def train_model(model, dataloader, criterion, optimizer, num_epochs=25, log_interval=10, checkpoint_interval=100):
    model.train()
    rmse_history = []
    batch_rmse_history = []
    best_rmse = float('inf')
    epochs_no_improve = 0
    start_epoch = 0

    checkpoint_path = f'{project_name}_model_checkpoint.pth'
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 출력 크기와 타겟 크기를 출력하여 확인합니다.
            print(f"Batch {batch_idx}: outputs size = {outputs.size()}, targets size = {targets.size()}")
            
            loss = criterion(outputs, targets.view(-1, 1))  # targets의 shape을 (batch_size, 1)로 맞춥니다.
            loss.backward()
            optimizer.step()

            batch_rmse = torch.sqrt(loss).item()
            epoch_losses.append(batch_rmse)
            batch_rmse_history.append(batch_rmse)

            if batch_idx % log_interval == 0:
                wandb.log({"epoch": epoch, "batch_idx": batch_idx, "batch_rmse": batch_rmse})
            
            # Save checkpoint every 100 batches
            if (batch_idx + 1) % checkpoint_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
                print(f"Checkpoint saved at batch {batch_idx+1} of epoch {epoch+1}")

        epoch_rmse = np.mean(epoch_losses)
        rmse_history.append(epoch_rmse)
        wandb.log({"epoch": epoch+1, "epoch_rmse": epoch_rmse})
        print(f"{datetime.now()} - Epoch [{epoch+1}/{num_epochs}], RMSE: {epoch_rmse:.4f}")

        # Save checkpoint at the end of each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at the end of epoch {epoch+1}")

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

print("Step 5: Training and Inference Functions - Completed")

# Step 6: Putting It All Together
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

train_data = pd.read_csv('/data/ephemeral/home/train.csv')
test_data = pd.read_csv('/data/ephemeral/home/test.csv')

# Apply the contract date processing before preprocessing
train_data = process_contract_date(train_data)
test_data = process_contract_date(test_data)

train_data = preprocess_data(train_data, include_target=True)
test_data = preprocess_data(test_data, include_target=False)

train_data = remove_outliers(train_data)

# One-hot encode the data
categorical_columns = ['시군구', '번지', '아파트명']
train_data, encoder = one_hot_encode(train_data, columns=categorical_columns)
test_data, _ = one_hot_encode(test_data, columns=categorical_columns, fitted_encoder=encoder)

# Align the test data with train data
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

# Print the shape of the final DataFrame before creating the dataset
print(f"Final train_data shape: {train_data.shape}")
print(f"Final test_data shape: {test_data.shape}")

train_dataset = RealEstateDataset(train_data, target_column='target')
test_dataset = RealEstateDataset(test_data)

print(f"{datetime.now()} - Create dataset instances...Done")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"{datetime.now()} - Create data loaders...Done")

model = SimpleResidualNetwork(input_dim=train_data.shape[1] - 1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if the model is on the GPU and print the status
if next(model.parameters()).is_cuda:
    print("Model is using GPU")
else:
    print("Model is using CPU")

trained_model, rmse_history, batch_rmse_history = train_model(model, train_dataloader, criterion, optimizer, num_epochs=num_epochs)

print(f"{datetime.now()} - Train model...Done")

plot_rmse(rmse_history, batch_rmse_history)

# Load the best model for prediction
checkpoint_path = f'{project_name}_model_checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from '{checkpoint_path}'")

predictions = predict(trained_model, test_dataloader)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
file_name = f'predictions_{timestamp}.csv'
predictions_df = pd.DataFrame(predictions, columns=['target'])
predictions_df.to_csv(file_name, index=False)

print(f"{datetime.now()} - Saved predictions to {file_name}...Done")

print("Step 6: Putting It All Together - Completed")
