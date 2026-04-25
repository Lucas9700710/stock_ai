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
    def __init__(self, data, lookback_window=30):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.lookback_window = lookback_window
        self.action_space = gym.spaces.Discrete(3) # 0:觀望, 1:買, 2:賣
        self.n_features = data.shape[1]
        
        # 原本是 lookback_window * self.n_features
        # 現在加上 3 個環境絕對狀態：現價、現金餘額、持倉數量
        self.state_dim = lookback_window * self.n_features + 3 
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        self.fee = 0.0001 
        self.tax = 0.0    
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
        
        # 時間序列標準化 (保留相對波動特徵)
        mean = window.mean(axis=0)
        std = window.std(axis=0) + 1e-8
        norm_window = (window - mean) / std
        flattened_obs = norm_window.flatten().astype(np.float32)
        
        # 提取現價 (window的最後一筆資料中，'Close' 位於 index 3)
        current_price = window[-1, 3]
        
        # 將絕對數值適度縮放，避免單一特徵數值過大 (例如 100000 的現金) 導致梯度爆炸
        # 這裡的除數可以根據你的標的物價格與初始資金微調
        extra_state = np.array([
            current_price / 1000.0,       # 縮放現價
            self.balance / 100000.0,      # 縮放可用現金 (相對於初始資金)
            self.shares / 100.0           # 縮放持倉量
        ], dtype=np.float32)
        
        # 將標準化後的歷史數據與絕對狀態合併
        return np.concatenate((flattened_obs, extra_state))
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
        
        # 獲取標的物本身的漲跌幅
        market_return = (price - self.data.iloc[self.current_step-1]['Close'].item()) / self.data.iloc[self.current_step-1]['Close'].item()
        # 獎勵 = (我的策略報酬 - 標的物報酬)
        reward = ((self.net_worth - prev_net_worth) / prev_net_worth) - market_return
        
        obs = self._get_obs() if not done else np.zeros(self.state_dim)
        return obs, reward, done, False, {}

# --- 3. 模型與 Double DQN Agent ---
class QNet(nn.Module):
    def __init__(self, seq_len=30, n_features=10, extra_dim=3, out_dim=3):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.extra_dim = extra_dim
        
        # LSTM 層：負責處理時間序列 (batch_first=True 代表輸入格式為 Batch 在前)
        self.lstm = nn.LSTM(
            input_size=n_features, 
            hidden_size=64,       # LSTM 輸出的特徵維度 
            num_layers=1,         # 1 層通常足夠，避免嚴重過擬合
            batch_first=True
        )
        
        # 決策層：結合 LSTM 的輸出與 3 維的絕對帳戶狀態
        self.fc = nn.Sequential(
            nn.Linear(64 + extra_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),      # 加入 Dropout 提高泛化能力
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        # x 的形狀會是 (batch_size, seq_len * n_features + extra_dim)
        
        # 1. 將輸入資料切分為「時間序列」與「帳戶絕對狀態」
        ts_len = self.seq_len * self.n_features
        ts_x = x[:, :ts_len]        # 取前 300 個數值 (30天 * 10特徵)
        extra_x = x[:, ts_len:]     # 取最後 3 個數值 (現價、現金、持倉)
        
        # 2. 將時間序列 Reshape 成 LSTM 需要的 3D 格式
        # 形狀變為: (batch_size, seq_len, n_features)
        ts_x = ts_x.view(-1, self.seq_len, self.n_features)
        
        # 3. 放入 LSTM 運算
        lstm_out, (hn, cn) = self.lstm(ts_x)
        
        # 我們只需要 LSTM 在「最後一個時間步 (last time step)」的輸出結果
        last_out = lstm_out[:, -1, :] 
        
        # 4. 將 LSTM 萃取出的特徵與帳戶狀態拼接
        combined = torch.cat((last_out, extra_x), dim=1)
        
        # 5. 通過全連接層輸出 3 個動作的 Q 值
        return self.fc(combined)

class DQNAgent:
    # 新增 seq_len 與 n_features 參數
    def __init__(self, state_dim, action_dim, seq_len=30, n_features=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9992
        self.lr = 0.0001 

        # 這裡改用新的參數初始化 QNet
        self.model = QNet(seq_len=seq_len, n_features=n_features, extra_dim=3, out_dim=action_dim).to(device)
        self.target_model = QNet(seq_len=seq_len, n_features=n_features, extra_dim=3, out_dim=action_dim).to(device)
        
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
    agent = DQNAgent(env.state_dim, 3, seq_len=env.lookback_window, n_features=env.n_features)

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