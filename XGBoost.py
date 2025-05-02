# ======================= Imports ===========================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt

# =================== Load & Preprocess =====================
train_df = pd.read_csv(r"C:\Users\pretam\Downloads\archive (8)\fraudTrain.csv")
train_df = train_df.drop(columns=['Unnamed: 0'], errors='ignore')
train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
train_df['hour'] = train_df['trans_date_trans_time'].dt.hour
train_df = train_df.drop(columns=['trans_date_trans_time'])

# Encode categorical variables
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].astype('category').cat.codes

# Scale numeric features
scaler = MinMaxScaler()
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])

print(' Preprocessing Done')

# =================== Split & Apply SMOTE ===================
X = train_df.drop('is_fraud', axis=1)
y = train_df['is_fraud']

# Train-test split before SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE only on training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f" After SMOTE: Train = {np.bincount(y_train_res)}, Test = {np.bincount(y_test)}")

# =================== Train XGBoost =========================
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_res, y_train_res)

# =================== Evaluate ==============================
y_pred = xgb_model.predict(X_test)

print("\n XGBoost Model Evaluation:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# =================== Feature Importance ====================
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain')
plt.title("Top 10 Important Features")
plt.grid(True)
plt.tight_layout()
plt.show()


#output

'''
 Preprocessing Done

 After SMOTE: Train = [1031335 1031335], Test = [257834   1501]

 XGBoost Model Evaluation:
[[255414   2420]
 [   284   1217]]
              precision    recall  f1-score   support

         0.0       1.00      0.99      0.99    257834
         1.0       0.33      0.81      0.47      1501

    accuracy                           0.99    259335
   macro avg       0.67      0.90      0.73    259335
weighted avg       1.00      0.99      0.99    259335

Accuracy: 0.9896 '''
