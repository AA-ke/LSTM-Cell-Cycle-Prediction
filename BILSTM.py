import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1️⃣ 加载数据
df = pd.read_csv('cell_features.csv')
df.columns = df.columns.str.strip()
df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
df['cell_id'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # 细胞编号
df = df.dropna(subset=['frame', 'cell_id'])

# 2️⃣ 计算特征
df['aspect_ratio'] = df['length'] / (df['area'] + 1e-6)
df['brightness_var'] = df['mean_intensity'].rolling(3).std().fillna(0)
df['motion_x'] = df.groupby('cell_id')['x'].diff().fillna(0)  # 每个细胞单独计算
df['motion_y'] = df.groupby('cell_id')['y'].diff().fillna(0)
df['speed'] = np.sqrt(df['motion_x'] ** 2 + df['motion_y'] ** 2)

df['perimeter'] = np.pi * (df['length'] + df['area'] / (df['length'] + 1e-6))
df['solidity'] = df['area'] / (df['perimeter'] + 1e-6)
df['eccentricity'] = np.sqrt(1 - (df['area'] / (df['length'] ** 2 + 1e-6)))
df['roundness'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2 + 1e-6)
df['intensity_variance'] = df.groupby('cell_id')['mean_intensity'].rolling(3).std().fillna(0).reset_index(level=0, drop=True)

# ✅ 计算 delta_frame（帧间隔）—— 仅在相同 cell_id 内计算
df['delta_frame'] = df.groupby('cell_id')['frame'].diff().fillna(1)  # 保证不跨细胞

# 3️⃣ 选择特征和目标
features = ['x', 'y', 'area', 'length', 'mean_intensity', 'aspect_ratio', 'brightness_var',
            'speed', 'eccentricity', 'solidity', 'roundness', 'intensity_variance']
target = 'delta_frame'

# 4️⃣ 归一化
scaler_X = MinMaxScaler()
df[features] = scaler_X.fit_transform(df[features])
scaler_y = MinMaxScaler()
df[[target]] = scaler_y.fit_transform(df[[target]])

# 5️⃣ 处理 LSTM 输入格式
X, y = [], []
window_size = 10

for cell_id, group in df.groupby('cell_id'):  # 按细胞编号分组，独立处理每个细胞的数据
    if len(group) > window_size:
        for i in range(window_size, len(group)):
            X.append(group[features].iloc[i - window_size:i].values)
            y.append(group[target].iloc[i])

X = np.array(X)
y = np.array(y)

# 6️⃣ 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 7️⃣ 构建 LSTM 网络
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(64, return_sequences=False)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),

    Dense(1, activation='relu')  # 约束 delta_frame > 0
])

# 8️⃣ 编译模型
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
model.summary()

# 9️⃣ 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 🔟 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 1️⃣1️⃣ 预测 & 反归一化
y_pred = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred)  # 反归一化 delta_frame
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# ✅ 计算真实预测的 frame 值
frame_start = df['frame'].iloc[len(X_train) + window_size]  # 测试集起始 frame
y_pred_frame = np.cumsum(y_pred) + frame_start  # 递增帧

# 计算均方误差
mse = np.mean((y_pred - y_test) ** 2)
print(f'Mean Squared Error: {mse}')

# 1️⃣2️⃣ 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(df['frame'].iloc[len(X_train) + window_size:len(X_train) + window_size + len(y_test)].values,
         label='True Frame')
plt.plot(y_pred_frame, label='Predicted Frame', linestyle='dashed')
plt.title('True vs Predicted Cell Division Frame')
plt.xlabel('Sample Index')
plt.ylabel('Frame')
plt.legend()
plt.show()
