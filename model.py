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
from datetime import date,timedelta
# --- 0. 系統設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "stock_dqn_model.pth"

# --- 1. 資料處理：加入更多特徵 ---
def get_cleaned_data(ticker, start, end):
    """從 yfinance 下載數據並加入技術指標"""
    df = yf.download(ticker, start=start, end=end, progress=False)
    # 技術指標：讓模型更有方向感
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).apply(lambda x: x[x>0].sum()/abs(x[x<0].sum()) if x[x<0].sum()!=0 else 10)))
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df = df.dropna()
    # 特徵欄位：開高低收、均線、RSI、報酬率 (共 8 維)
    return df[['Open', 'High', 'Low', 'Close', 'MA5', 'MA20', 'RSI', 'Log_Ret']]

# --- 2. 交易環境：考慮手續費與 30 天視窗 ---
class StockTradingEnv(gym.Env):
    def __init__(self, data, lookback_window=30):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.lookback_window = lookback_window
        self.action_space = gym.spaces.Discrete(3) # 0:觀望, 1:買, 2:賣
        self.state_dim = lookback_window * 8
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        self.fee = 0.001425 # 台股手續費
        self.tax = 0.003    # 證交稅
        self.reset()

    def reset(self, seed=None):
        '''重設還境，初始資金 10 萬，無持股'''
        super().reset(seed=seed)
        self.balance = 100000.0
        self.shares = 0
        self.net_worth = 100000.0
        self.current_step = self.lookback_window
        return self._get_obs(), {}

    def _get_obs(self):
        '''抓取過去 30 天數據'''
        window = self.data.iloc[self.current_step - self.lookback_window : self.current_step].values
        # Z-Score 正規化：讓模型在不同股價水準下都能學習
        mean = window.mean(axis=0)
        std = window.std(axis=0) + 1e-8
        norm_window = (window - mean) / std
        return norm_window.flatten().astype(np.float32)

    def step(self, action):
        "''執行動作，計算獎勵，更新狀態'''"
        price = self.data.iloc[self.current_step]['Close'].item()
        prev_net_worth = self.net_worth

        # 動作邏輯
        if action == 1 and self.balance > price: # Buy
            can_buy = int(self.balance / (price * (1 + self.fee)))
            if can_buy > 0:
                self.balance -= can_buy * price * (1 + self.fee)
                self.shares += can_buy
        elif action == 2 and self.shares > 0: # Sell
            self.balance += self.shares * price * (1 - self.fee - self.tax)
            self.shares = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # 更新淨資產 (使用第 31 天的收盤價計算損益)
        self.net_worth = self.balance + (self.shares * price)
        reward = (self.net_worth - prev_net_worth) / prev_net_worth # 獲利百分比作為獎勵
        
        obs = self._get_obs() if not done else np.zeros(self.state_dim)
        return obs, reward, done, False, {}

# --- 3. 模型與 Agent ---
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
        """DQN Agent 初始化"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.997
        
        
        # 1. 先建立模型實體
        self.model = QNet(state_dim, action_dim).to(device)
        self.target_model = QNet(state_dim, action_dim).to(device)
        
        # 2. 檢查檔案是否存在，存在才載入權重
        if os.path.exists(MODEL_PATH):
            try:
                state_dict = torch.load(MODEL_PATH, map_location=device)
                self.model.load_state_dict(state_dict)
                print(f"成功從 {MODEL_PATH} 載入預訓練權重！")
            except Exception as e:
                print(f"載入權重時發生錯誤（可能是模型結構不符）: {e}")
        else:
            print("找不到模型檔案，將從隨機權重開始訓練。")
        # ------------------

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.update_target()

    def update_target(self):
        "''將主模型權重複製到目標模型'''"
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, train=True):
        "''根據 epsilon-greedy 策略選擇動作'''"
        if train and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.model(s).argmax().item()

    def train(self, batch_size):
        "''從記憶庫中抽取批次進行訓練'''"
        if len(self.memory) < batch_size: return 
        batch = random.sample(self.memory, batch_size)
        s, a, r, ns, d = zip(*batch)
        
        s = torch.FloatTensor(np.array(s)).to(device)
        a = torch.LongTensor(a).unsqueeze(1).to(device)
        r = torch.FloatTensor(r).to(device)
        ns = torch.FloatTensor(np.array(ns)).to(device)
        d = torch.FloatTensor(d).to(device)

        q_val = self.model(s).gather(1, a).squeeze()
        next_q = self.target_model(ns).max(1)[0]
        target = r + (1 - d) * self.gamma * next_q
        
        loss = nn.MSELoss()(q_val, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

# --- 4. 執行與分析報告 ---
def run_simulation():
    ticker = "GOOG"
    data = get_cleaned_data(ticker, "2010-01-01", "2022-04-26")
    env = StockTradingEnv(data)
    agent = DQNAgent(env.state_dim, 3)

    # 訓練階段
    EPISODES = 10
    print(f"開始優化 {ticker} 的交易策略...")
    for e in range(EPISODES):
        state, _ = env.reset()
        for t in range(len(data)-32):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            if t % 10 == 0: agent.train(16) # 每 n 步訓練一次
            if done: break
        agent.update_target()
        print(f"Episode {e+1}/{EPISODES} | Net Worth: {env.net_worth:.2f} | Eps: {agent.epsilon:.2f}")

    # 儲存模型
    torch.save(agent.model.state_dict(), MODEL_PATH)
    print(f"模型已成功儲存於 {MODEL_PATH}")

    # 測試與績效分析
    data = get_cleaned_data(ticker, "2022-01-01",date.today().strftime('%Y-%m-%d'))
    env = StockTradingEnv(data)
    state, _ = env.reset()
    history = []
    for n in range(len(data)-32):
        action = agent.act(state, train=False)
        state, _, done, _, _ = env.step(action)
        print(f"Step: {n+1} | Action: {['Hold', 'Buy', 'Sell'][action]} | Net Worth: {env.net_worth:.2f}") # 每一步都印出動作與淨資產
        history.append(env.net_worth)
        if done: break

    # 計算績效指標
    history = np.array(history)
    returns = np.diff(history) / history[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    max_dd = (history / np.maximum.accumulate(history) - 1).min() * 100

    print(f"\n--- 最終績效報告 ---")
    print(f"最終資產: {history[-1]:.2f}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_dd:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(history, label='AI Strategy', color='royalblue')
    plt.title(f"AI Trading Performance: {ticker}")
    plt.ylabel("Net Worth (TWD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("final_performance.png")
    #plt.show()
    print("績效圖已儲存為 final_performance.png")
    plt.close()

if __name__ == "__main__":
    run_simulation()