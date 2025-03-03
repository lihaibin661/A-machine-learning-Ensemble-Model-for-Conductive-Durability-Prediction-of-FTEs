import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# -------------------------
# 1. 数据加载与预处理
# -------------------------
# 加载数据
data = pd.read_csv('更新线性回归600.csv')

# 移除完全为空的列
data = data.dropna(axis=1, how='all')

# 检查并填充缺失值（使用均值填充）
imputer = SimpleImputer(strategy='mean')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 确保所有缺失值被处理
if data_filled.isnull().sum().sum() > 0:
    print("数据仍有缺失值，请检查数据处理部分。")
else:
    print("缺失值已成功填充。")

# 选择输入特征和目标变量（假设目标为最后一列）
X = data_filled.iloc[:, :-1]  # 输入特征为前几列
y = data_filled.iloc[:, -1]   # 目标变量为最后一列（例如 Square resistance）

# -------------------------
# 2. 特征标准化与多项式扩展
# -------------------------
# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用二阶多项式扩展特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# -------------------------
# 3. 划分训练集和测试集
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# -------------------------
# 4. 搭建与训练 TensorFlow 模型
# -------------------------
# 构建模型（含 L2 正则化）
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],),
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1)
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# 训练模型 100 个 Epoch，并保留验证集
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

# -------------------------
# 5. 绘制 Epoch-Loss 曲线
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')

# 在指定 Epoch（10, 50, 100）处标注 loss 值
for epoch in [10, 50, 100]:
    if epoch <= len(history.history['loss']):
        train_loss_value = history.history['loss'][epoch - 1]
        val_loss_value = history.history['val_loss'][epoch - 1]
        plt.scatter(epoch, train_loss_value, color='blue', s=50)
        plt.text(epoch, train_loss_value, f'{train_loss_value:.4f}', fontsize=18, fontname='Times New Roman', 
                 color='blue', ha='right')
        plt.scatter(epoch, val_loss_value, color='orange', s=50)
        plt.text(epoch, val_loss_value, f'{val_loss_value:.4f}', fontsize=18, fontname='Times New Roman', 
                 color='orange', ha='left')

plt.xlabel('Epoch', fontsize=18, fontname='Times New Roman')
plt.ylabel('Loss (MSE)', fontsize=18, fontname='Times New Roman')
plt.title('Epoch-Loss Curve', fontsize=24, fontweight='bold', fontname='Times New Roman')
plt.legend(fontsize=18, loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 6. 模型预测与评估
# -------------------------
# 使用模型在测试集上进行预测
y_pred = model.predict(X_test).flatten()

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
mb = np.mean(y_pred - y_test)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Bias (MB):", mb)
print("Mean Absolute Percentage Error (MAPE):", mape)

# 绘制实际值与预测值的散点图（添加理想 y=x 参考线）
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label='Data Points', color='blue')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')
plt.xlabel('Actual Values', fontsize=18, fontname='Times New Roman')
plt.ylabel('Predicted Values', fontsize=18, fontname='Times New Roman')
plt.title('Actual vs Predicted Values', fontsize=24, fontweight='bold', fontname='Times New Roman')
plt.legend(fontsize=18, loc='lower right')
plt.text(y_test.min(), y_test.max() * 0.9, f'Scatter Spread (MSE): {mse:.4f}', fontsize=18, 
         fontname='Times New Roman', color='black')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 7. 计算并绘制皮尔逊相关系数热力图
# -------------------------
# 计算原始输入特征（标准化后）的皮尔逊相关系数与目标之间的相关性
# 这里使用标准化后的 X（未做多项式扩展），便于理解各原始特征与目标的关系
feature_names = X.columns  # 输入特征的名称
data_features = pd.DataFrame(X_scaled, columns=feature_names)
data_features['Target'] = y.values

# 计算相关矩阵
pearson_corr = data_features.corr()

# 这里我们只关注每个输入特征与 Target 之间的相关系数
pearson_input_target = pearson_corr[['Target']].drop('Target')

# 绘制热力图（二维形式展示相关系数值）
plt.figure(figsize=(8, len(pearson_input_target)*0.8))
plt.imshow(pearson_input_target, cmap='coolwarm', interpolation='nearest', aspect='auto')
plt.colorbar()
plt.xticks([0], ['Target'], fontsize=18, fontname='Times New Roman')
plt.yticks(range(len(pearson_input_target.index)), pearson_input_target.index, fontsize=18, fontname='Times New Roman')
plt.title('Pearson Correlation (Input vs Target)', fontsize=24, fontweight='bold', fontname='Times New Roman')

# 添加相关系数数值标签
for i in range(len(pearson_input_target.index)):
    plt.text(0, i, f'{pearson_input_target.iloc[i, 0]:.2f}', ha='center', va='center', 
             color='black', fontsize=18, fontname='Times New Roman')
plt.tight_layout()
plt.show()
