# ================== Load & Preprocess Full Data ==================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r"C:\Users\pretam\Downloads\archive (8)\fraudTrain.csv")
df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df.drop(columns=['trans_date_trans_time'], inplace=True)
df = df.apply(lambda col: col.astype('category').cat.codes if col.dtypes == 'object' else col)

# Scale numeric features
scaler = MinMaxScaler()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = scaler.fit_transform(df[num_cols])

print("✅ Preprocessing done on full dataset")

# ========== Sample Realistic Subset ==========
df_majority = df[df.is_fraud == 0].sample(n=10000, random_state=42)
df_minority = df[df.is_fraud == 1]  # keep all fraud
df_subset = pd.concat([df_majority, df_minority])

X = df_subset.drop('is_fraud', axis=1)
y = df_subset['is_fraud']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# ========== Train-Test Split ==========
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# ========== Define Gym Environment ==========
import gym

class FraudEnv(gym.Env):
    def __init__(self, X, y):
        self.X = X.astype(np.float32).reset_index(drop=True)
        self.y = y.astype(np.int32).reset_index(drop=True)
        self.state = 0
        self.done = False

    def reset(self):
        self.state = 0
        self.done = False
        return self.X.iloc[self.state].to_numpy()

    def step(self, action):
        reward = 1 if action == self.y[self.state] else -1
        self.state += 1
        if self.state >= len(self.y):
            self.done = True
            next_state = np.zeros_like(self.X.iloc[0].to_numpy())
        else:
            next_state = self.X.iloc[self.state].to_numpy()
        return next_state, reward, self.done, {}

train_env = FraudEnv(X_train, y_train)
test_env = FraudEnv(X_test, y_test)

# ========== Build DQN Model ==========
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from collections import deque
import random
import os

dqn_model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="linear")
])
dqn_model.compile(optimizer=optimizers.Adam(0.001), loss="mse")

# ========== Train DQN (Faster Config) ==========
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99
batch_size = 32
memory = deque(maxlen=5000)
episodes = 5  # ✅ Reduce for speed
reward_per_episode = []

for episode in range(1, episodes + 1):
    state = train_env.reset()
    total_reward = 0
    while True:
        action = np.random.choice([0, 1]) if np.random.rand() < epsilon else np.argmax(dqn_model.predict(state.reshape(1, -1), verbose=0))
        next_state, reward, done, _ = train_env.step(action)
        memory.append((state, action, reward, next_state, done))
        total_reward += reward
        state = next_state

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            q_vals = dqn_model.predict(np.array(states), verbose=0)
            next_q_vals = dqn_model.predict(np.array(next_states), verbose=0)
            targets = rewards + gamma * np.max(next_q_vals, axis=1) * (~np.array(dones))
            q_vals[np.arange(batch_size), actions] = targets
            dqn_model.fit(np.array(states), q_vals, epochs=1, verbose=0, batch_size=batch_size)

        if done:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    reward_per_episode.append(total_reward)
    print(f"✅ Episode {episode} | Reward: {total_reward} | Epsilon: {epsilon:.3f}")

# Save Model
os.makedirs("saved_model", exist_ok=True)
dqn_model.save("saved_model/dqn_realistic_subset.keras")

# ========== Evaluation ==========
state = test_env.reset()
y_true, y_pred = [], []
total_test_reward = 0

while True:
    action = np.argmax(dqn_model.predict(state.reshape(1, -1), verbose=0))
    y_pred.append(action)
    y_true.append(test_env.y[test_env.state])
    next_state, reward, done, _ = test_env.step(action)
    total_test_reward += reward
    state = next_state
    if done:
        break

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("\n✅ DQN Evaluation (realistic subset):")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")


# OUTPUT

'''
 Episode 1 | Reward: 34 | Epsilon: 0.990
✅ Episode 2 | Reward: 76 | Epsilon: 0.980
✅ Episode 3 | Reward: -100 | Epsilon: 0.970
✅ Episode 4 | Reward: -46 | Epsilon: 0.961
✅ Episode 5 | Reward: 122 | Epsilon: 0.951

✅ DQN Evaluation (realistic subset):
[[   0 2000]
 [   0 2000]]

             precision    recall  f1-score   support

           0       0.00      0.00      0.00      2000
           1       0.50      1.00      0.67      2000

    accuracy                           0.50      4000
   macro avg       0.25      0.50      0.33      4000
weighted avg       0.25      0.50      0.33      4000

Accuracy: 0.5000
'''