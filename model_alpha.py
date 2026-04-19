import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import random
import os
from collections import deque
from datetime import date

# --- 0. 系統設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "stock_ddqn_model.pth"

# --- 1. 資料處理：加入 10 維特徵 ---
def get_cleaned_data(ticker, start, end):
    """從 yfinance 下載數據並加入技術指標"""
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    # 處理 yfinance 可能產生的 Multi-Index 欄位
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 技術指標
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Change'] = df['Volume'].pct_change()
    df['BIAS'] = (df['Close'] - df['MA20']) / df['MA20']
    
    df = df.dropna()
    # 特徵欄位共 10 維
    features = ['Open', 'High', 'Low', 'Close', 'MA5', 'MA20', 'RSI', 'Log_Ret', 'Vol_Change', 'BIAS']
    return df[features]

# --- 2. 交易環境 ---
class StockTradingEnv(gym.Env):
    def __init__(self, data, lookback_window=60):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.lookback_window = lookback_window
        self.action_space = gym.spaces.Discrete(3) # 0:觀望, 1:買, 2:賣
        self.n_features = data.shape[1]
        self.state_dim = lookback_window * self.n_features
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        self.fee = 0.0001 # 模擬美股佣金
        self.tax = 0.0    # 美股無證交稅
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = 100000.0
        self.shares = 0
        self.net_worth = 100000.0
        self.current_step = self.lookback_window
        return self._get_obs(), {}

    def _get_obs(self):
        window = self.data.iloc[self.current_step - self.lookback_window : self.current_step].values
        mean = window.mean(axis=0)
        std = window.std(axis=0) + 1e-8
        norm_window = (window - mean) / std
        return norm_window.flatten().astype(np.float32)

    def step(self, action):
        price = self.data.iloc[self.current_step]['Close'].item()
        prev_net_worth = self.net_worth

        # 動作邏輯
        if action == 1 and self.balance > price: # Buy
            can_buy = int(self.balance / (price * (1 + self.fee)))
            if can_buy > 0:
                self.balance -= can_buy * price * (1 + self.fee)
                self.shares += can_buy
        elif action == 2 and self.shares > 0: # Sell
            self.balance += self.shares * price * (1 - self.fee)
            self.shares = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        self.net_worth = self.balance + (self.shares * price)
        
        # 獎勵設計：包含手續費懲罰
        reward = (self.net_worth - prev_net_worth) / prev_net_worth
        if action != 0:
            reward += 0.0005
        
        obs = self._get_obs() if not done else np.zeros(self.state_dim)
        return obs, reward, done, False, {}

# --- 3. 模型與 Double DQN Agent ---
class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x): return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9992
        self.lr = 0.0001 # 較低的學習率

        self.model = QNet(state_dim, action_dim).to(device)
        self.target_model = QNet(state_dim, action_dim).to(device)
        
        if os.path.exists(MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                print(f"成功載入預訓練權重！")
            except:
                print("模型結構變更，從新開始訓練。")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, train=True):
        if train and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.model(s).argmax().item()

    def train(self, batch_size):
        if len(self.memory) < batch_size: return 
        batch = random.sample(self.memory, batch_size)
        s, a, r, ns, d = zip(*batch)
        
        s = torch.FloatTensor(np.array(s)).to(device)
        a = torch.LongTensor(a).unsqueeze(1).to(device)
        r = torch.FloatTensor(r).to(device)
        ns = torch.FloatTensor(np.array(ns)).to(device)
        d = torch.FloatTensor(d).to(device)

        # Double DQN 核心邏輯
        with torch.no_grad():
            next_action = self.model(ns).argmax(1).unsqueeze(1)
            next_q = self.target_model(ns).gather(1, next_action).squeeze()
            target = r + (1 - d) * self.gamma * next_q
        
        q_val = self.model(s).gather(1, a).squeeze()
        loss = nn.MSELoss()(q_val, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- 4. 執行與分析 ---
def run_simulation():
    ticker = "SNPS"
    # 訓練階段
    train_data = get_cleaned_data(ticker, "2020-01-01", "2023-12-31")
    env = StockTradingEnv(train_data)
    agent = DQNAgent(env.state_dim, 3)

    EPISODES = 100
    print(f"開始訓練 {ticker} 策略 (Double DQN)...")
    for e in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            agent.train(128)
            if done: break
        
        agent.update_target()
        if (e + 1) % 1 == 0:
            print(f"Episode {e+1}/{EPISODES} | Net Worth: {env.net_worth:.2f} | Eps: {agent.epsilon:.2f}")

    torch.save(agent.model.state_dict(), MODEL_PATH)

    # 測試階段
    print("\n--- 開始 2024+ 樣本外測試 ---")
    test_data = get_cleaned_data(ticker, "2024-01-01", date.today().strftime('%Y-%m-%d'))
    test_env = StockTradingEnv(test_data)
    state, _ = test_env.reset()
    history = []
    
    while True:
        action = agent.act(state, train=False)
        state, _, done, _, _ = test_env.step(action)
        history.append(test_env.net_worth)
        if done: break

    history = np.array(history)
    returns = np.diff(history) / history[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    max_dd = (history / np.maximum.accumulate(history) - 1).min() * 100

    print(f"\n--- 最終績效報告 ---")
    print(f"最終資產: {history[-1]:.2f}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_dd:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(history, label='Double DQN Strategy', color='forestgreen')
    plt.title(f"AI Trading: {ticker} (2024-Present)")
    plt.ylabel("Net Worth (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("final_performance.png")
    plt.close()

if __name__ == "__main__":
    run_simulation()