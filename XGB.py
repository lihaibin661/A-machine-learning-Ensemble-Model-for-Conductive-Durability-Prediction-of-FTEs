import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import shap

##########################
# 1. 数据加载与预处理
##########################

# 1.1 线性回归数据（用于构建 LR 基模型）
linear_data = pd.read_csv('更新线性回归600.csv')
# 假设选择最后三列作为输入特征，目标取第 7 列（索引 6）
X_linear = linear_data.iloc[:, -3:]
y_linear = linear_data.iloc[:, 6]

# 填充缺失值、标准化、生成二阶多项式特征
imputer = SimpleImputer(strategy='mean')
X_linear_imputed = imputer.fit_transform(X_linear)
scaler = StandardScaler()
X_linear_scaled = scaler.fit_transform(X_linear_imputed)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_linear_poly = poly.fit_transform(X_linear_scaled)

X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
    X_linear_poly, y_linear, test_size=0.2, random_state=42
)

# 1.2 CNN 和 RF 数据（用于构建 CNN 和 RF 基模型）
cnn_data = pd.read_csv('E:/江流 小论文 模型/model/CNN.PY/600CNN5.13.csv')
# 拆分 'ZTO/Ag' 列为两个特征
cnn_data[['ZTO', 'Ag']] = cnn_data['ZTO/Ag'].str.split(',', expand=True)
cnn_data['ZTO'] = pd.to_numeric(cnn_data['ZTO'])
cnn_data['Ag'] = pd.to_numeric(cnn_data['Ag'])

# 独热编码 'ZTO' 和 'Ag'
encoder = OneHotEncoder(sparse_output=False)
encoded_thickness = encoder.fit_transform(cnn_data[['ZTO', 'Ag']])
# 拼接其他特征
X_cnn = np.hstack((encoded_thickness, cnn_data[['Number of Bending', 'Storage time']].values))
y_cnn = cnn_data['Square resistance'].values

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn, y_cnn, test_size=0.2, random_state=42
)

##########################
# 2. 基模型训练
##########################

# 2.1 LR 模型（使用简单全连接网络模拟线性回归）
linear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_linear.shape[1],)),
    tf.keras.layers.Dense(1)
])
linear_model.compile(optimizer='adam', loss='mean_squared_error')
linear_history = linear_model.fit(X_train_linear, y_train_linear, epochs=100, validation_split=0.2, verbose=1)

# 训练集和测试集上的预测
linear_pred_train = linear_model.predict(X_train_linear).flatten()
linear_pred_test = linear_model.predict(X_test_linear).flatten()

# 2.2 CNN 模型
# CNN 需要将输入 reshape 成 (samples, features, 1)
X_train_cnn_reshaped = X_train_cnn.reshape(X_train_cnn.shape[0], X_train_cnn.shape[1], 1)
X_test_cnn_reshaped = X_test_cnn.reshape(X_test_cnn.shape[0], X_test_cnn.shape[1], 1)

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train_cnn_reshaped.shape[1], 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
])
cnn_model.compile(optimizer='adam', loss='mean_squared_error')
cnn_history = cnn_model.fit(X_train_cnn_reshaped, y_train_cnn, epochs=100, validation_split=0.2, verbose=1)

cnn_pred_train = cnn_model.predict(X_train_cnn_reshaped).flatten()
cnn_pred_test = cnn_model.predict(X_test_cnn_reshaped).flatten()

# 2.3 随机森林模型（使用 CNN 数据作为输入）
# 这里直接使用 X_cnn 拼接后的数据，不进行 reshape
rf_model = RandomForestRegressor(random_state=42)
# 使用与 CNN 数据相同的划分
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_cnn, y_cnn, test_size=0.2, random_state=42
)
rf_model.fit(X_train_rf, y_train_rf)
rf_pred_train = rf_model.predict(X_train_rf)
rf_pred_test = rf_model.predict(X_test_rf)

##########################
# 3. 堆叠模型（XGBoost 元模型）
##########################
# 构造元特征：将三个基模型在训练集的预测堆叠起来
meta_X_train = np.column_stack((linear_pred_train, cnn_pred_train, rf_pred_train))
meta_X_test = np.column_stack((linear_pred_test, cnn_pred_test, rf_pred_test))
# 假设 y_train_linear 为三者共同的目标（注意：确保数据对应，否则请使用统一的目标）
meta_y_train = y_train_linear

# 使用部分测试集的目标作为验证（这里用 X_test_linear 对应的 y_test_linear 作为验证）
# 训练 XGBoost 元模型，并记录训练过程的评估指标（RMSE）
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
eval_set = [(meta_X_train, meta_y_train), (meta_X_test, y_test_linear)]
xgb_model.fit(meta_X_train, meta_y_train, eval_metric="rmse", eval_set=eval_set, verbose=True)

# 提取 XGBoost 训练过程的评估结果，绘制 Epoch-Loss 图
results = xgb_model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

plt.figure(figsize=(10,6))
plt.plot(x_axis, results['validation_0']['rmse'], label='Train RMSE', marker='o')
plt.plot(x_axis, results['validation_1']['rmse'], label='Validation RMSE', marker='x')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('XGBoost RMSE Loss Curve (Stacking Meta Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 使用 XGBoost 元模型在测试集上预测
meta_y_pred = xgb_model.predict(meta_X_test)

# 计算评估指标
mse_meta = mean_squared_error(y_test_linear, meta_y_pred)
rmse_meta = np.sqrt(mse_meta)
mae_meta = mean_absolute_error(y_test_linear, meta_y_pred)
mb_meta = np.mean(y_test_linear - meta_y_pred)
mape_meta = np.mean(np.abs((y_test_linear - meta_y_pred) / y_test_linear)) * 100
r2_meta = r2_score(y_test_linear, meta_y_pred)

print("Meta Model Evaluation:")
print("MAE:", mae_meta)
print("RMSE:", rmse_meta)
print("MB:", mb_meta)
print("MAPE:", mape_meta)
print("R2:", r2_meta)

##########################
# 4. 绘制散点图（实际 vs 预测）
##########################
plt.figure(figsize=(10,6))
plt.scatter(y_test_linear, meta_y_pred, alpha=0.6, color='blue', label='Meta Predictions')
min_val = min(y_test_linear.min(), meta_y_pred.min())
max_val = max(y_test_linear.max(), meta_y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')
plt.xlabel('Actual Values', fontsize=18, fontname='Times New Roman')
plt.ylabel('Predicted Values', fontsize=18, fontname='Times New Roman')
plt.title('Stacking Meta Model: Actual vs Predicted', fontsize=24, fontweight='bold', fontname='Times New Roman')
plt.legend(fontsize=18, loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

##########################
# 5. （可选）使用 SHAP 分析 XGBoost 元模型
##########################
explainer = shap.Explainer(xgb_model)
shap_values = explainer(meta_X_test)
feature_names = ['LR_Pred', 'CNN_Pred', 'RF_Pred']
plt.figure(figsize=(10,8))
shap.summary_plot(shap_values, meta_X_test, feature_names=feature_names, plot_type="bar", show=False)
plt.title('SHAP Feature Contributions (Meta Model)', fontsize=20, fontweight='bold', fontname='Times New Roman')
plt.tight_layout()
plt.show()
