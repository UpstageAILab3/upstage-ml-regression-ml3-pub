import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
from scipy import stats  # stats 모듈 임포트

# Ensure the torch is using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Data Preprocessing Functions

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
            else:
                mode_value = df[column].mode()[0]
                df[column].fillna(mode_value, inplace=True)
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = radius * c
    return distance

def process_contract_date(data):
    data["계약연도"] = data["계약년월"].astype(str).str[:4]
    data["계약월"] = data["계약년월"].astype(str).str[4:]
    if '계약일' not in data.columns:
        data['계약일'] = '01'
    else:
        data['계약일'] = data['계약일'].apply(lambda x: '01' if pd.isnull(x) or not (1 <= int(x) <= 31) else f'{int(x):02d}')
    data["계약년월일"] = pd.to_datetime(data["계약년월"].astype(str) + data["계약일"].astype(str), errors='coerce')
    return data

def preprocess_data(df, include_target=False):
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

    df = df.drop(columns=['시', '구', '동', '대장아파트_거리','대장_좌표X', '대장_좌표Y'])
    
    # Remove outliers based on 계약년월일
    if include_target:
        mask = (df['계약년월일'] >= '2021-01-01') & (df['계약년월일'] <= '2022-01-31')
        z_scores = np.abs(stats.zscore(df.loc[mask, 'target']))
        df = df[~((z_scores >= 3) & mask)]
    
    # Drop 계약년월일 column after processing
    df = df.drop(columns=['계약년월일'])

    return df

def vectorize_text_columns(df, vectorizer=None, fit=True):
    text_columns = ['시군구', '아파트명']
    
    if fit:
        vectorizer = TfidfVectorizer()
        vectorized_data = vectorizer.fit_transform(df[text_columns].apply(lambda x: ' '.join(x), axis=1))
    else:
        vectorized_data = vectorizer.transform(df[text_columns].apply(lambda x: ' '.join(x), axis=1))
    
    vectorized_df = pd.DataFrame.sparse.from_spmatrix(vectorized_data, columns=vectorizer.get_feature_names_out())
    df = df.drop(columns=text_columns).reset_index(drop=True)
    vectorized_df = vectorized_df.reset_index(drop=True)
    df = pd.concat([df, vectorized_df], axis=1)
    
    return df, vectorizer

# Step 3: Custom Dataset Class for PyTorch
class RealEstateDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.apply(pd.to_numeric, errors='coerce').fillna(0)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataframe.iloc[idx]
        features = sample.values
        return torch.tensor(features, dtype=torch.float32)

# Step 4: Define the PyTorch Model
class SimpleResidualNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SimpleResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.residual = nn.Linear(input_dim, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.hidden_activation = nn.PReLU()

    def forward(self, x):
        out = self.hidden_activation(self.fc1(x))
        out = self.hidden_activation(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out) + self.residual(x)
        return out

# Step 6: Prediction Only

# Define selected features
selected_features = ['시군구', '번지', '아파트명', '전용면적(㎡)', '계약년월', '건축년도', '좌표X', '좌표Y']

# Define lead_house dictionary
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

# Load training data to fit vectorizer and scaler
train_data = pd.read_csv('/data/ephemeral/home/train.csv', low_memory=False)
train_data = process_contract_date(train_data)
train_data = preprocess_data(train_data, include_target=True)

# Initialize and fit the scaler on training data
scaler = MinMaxScaler()
train_data['target'] = scaler.fit_transform(train_data[['target']])

# Vectorize the text columns using TfidfVectorizer
train_data, vectorizer = vectorize_text_columns(train_data, fit=True)

# Load test data
test_data = pd.read_csv('/data/ephemeral/home/test.csv', low_memory=False)
test_data = process_contract_date(test_data)
test_data = preprocess_data(test_data, include_target=False)
test_data, _ = vectorize_text_columns(test_data, vectorizer=vectorizer, fit=False)

# Create test dataset and dataloader
test_dataset = RealEstateDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model
model = SimpleResidualNetwork(input_dim=test_data.shape[1]).to(device)
checkpoint_path = 'real-estate-prices-trail-7th_model_checkpoint.pth'  # Update with your checkpoint path

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Model loaded from '{checkpoint_path}'")

# Predict using the loaded model
model.eval()
predictions = []
with torch.no_grad():
    for inputs in test_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())

# Inverse transform the target values
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Save predictions
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
file_name = f'predictions_{timestamp}.csv'
predictions_df = pd.DataFrame(predictions, columns=['target'])
predictions_df.to_csv(file_name, index=False)

print(f"Predictions saved to {file_name}")
