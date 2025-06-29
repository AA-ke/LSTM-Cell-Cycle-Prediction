import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1️⃣ 读取数据
df = pd.read_csv('cell_features.csv')

# 清理列名（去掉空格或特殊字符）
df.columns = df.columns.str.strip()

# 确保 'frame' 列是数值类型
df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
df = df.dropna(subset=['frame'])  # 删除缺失值

# 选择特征和目标
features = ['x', 'y', 'area', 'length', 'mean_intensity']
target = 'frame'

# 确保所有特征列为数值类型
for feature in features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
df = df.dropna()  # 移除可能的 NaN 值

# 归一化特征
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# 2️⃣ 处理数据为 GRU 输入格式
X, y = [], []
window_size = 20  # GRU 时间窗口

for i in range(window_size, len(df)):
    X.append(df[features].iloc[i - window_size:i].values)  # 过去窗口大小的数据
    y.append(df[target].iloc[i])  # 当前帧

X = np.array(X)
y = np.array(y)

# 3️⃣ 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4️⃣ 构建 GRU 模型
model = Sequential([
    GRU(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    GRU(128, return_sequences=True),
    GRU(64, return_sequences=False),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)  # 确保输出 1 维
])

# 5️⃣ 编译模型
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# 6️⃣ 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 7️⃣ 绘制训练曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 8️⃣ 预测测试集
y_pred = model.predict(X_test).flatten()  # 修正形状，确保与 y_test 维度匹配

# 计算均方误差 (MSE)
mse = np.mean((y_pred - y_test) ** 2)
print(f'Mean Squared Error: {mse}')

# 9️⃣ 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values')
plt.plot(y_pred, label='Predicted Values')
plt.title('True vs Predicted Cell Division Time')
plt.xlabel('Sample Index')
plt.ylabel('Division Time (Frame)')
plt.legend()
plt.show()
