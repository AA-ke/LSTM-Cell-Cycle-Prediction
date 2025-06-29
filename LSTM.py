import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1️⃣ 加载数据
df = pd.read_csv('cell_features.csv')
df.columns = df.columns.str.strip()  # 清理列名
df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
df = df.dropna(subset=['frame'])  # 删除 NaN

# 2️⃣ 选择特征和目标
features = ['x', 'y', 'area', 'length', 'mean_intensity']
target = 'frame'

# 3️⃣ 归一化数据
scaler_X = MinMaxScaler()
df[features] = scaler_X.fit_transform(df[features])

scaler_y = MinMaxScaler()
df[['frame']] = scaler_y.fit_transform(df[['frame']])

# 4️⃣ 处理 LSTM 输入格式
X, y = [], []
window_size = 50  # 窗口大小

for i in range(window_size, len(df)):
    X.append(df[features].iloc[i - window_size:i].values)
    y.append(df[target].iloc[i])

X = np.array(X)
y = np.array(y)

# 5️⃣ 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 6️⃣ 搭建 LSTM 网络
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # 预测单个值
])

# 7️⃣ 编译模型
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
model.summary()

# 8️⃣ 训练模型
history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test))

# 9️⃣ 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 🔟 预测 & 反归一化
y_pred = model.predict(X_test)

# **确保 y_pred 和 y_test 都被反归一化**
y_pred = scaler_y.inverse_transform(y_pred)  # 预测值反归一化
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))  # 真实值反归一化

# 计算均方误差
mse = np.mean((y_pred - y_test) ** 2)
print(f'Mean Squared Error: {mse}')

# 1️⃣1️⃣ 绘制预测 vs 真实值
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values')
plt.plot(y_pred, label='Predicted Values')
plt.title('True vs Predicted Cell Division Time')
plt.xlabel('Sample Index')
plt.ylabel('Division Time (Frame)')
plt.legend()
plt.show()
