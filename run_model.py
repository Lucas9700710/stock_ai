import os
import torch
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import date,timedelta
import torch.nn as nn


# --- 1. 模型與環境設定 ---
class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x): return self.fc(x)

def get_cleaned_data(ticker, start, end=None):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).apply(
        lambda x: x[x>0].sum()/abs(x[x<0].sum()) if x[x<0].sum()!=0 else 10)))
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    return df.dropna()[['Open', 'High', 'Low', 'Close', 'MA5', 'MA20', 'RSI', 'Log_Ret']]

# --- 2. 每日自動執行函式 ---
def run_daily_simulation(model, ticker="2330.TW", simulate_date=None):
    history_file = "trade_history.npy"
    status_file = "account_status.npy"
    
    # 決定抓取數據的結束日：yfinance 的 end 是 exclusive (不包含該日)
    # 若要獲取 A 日的收盤價，end 必須設為 A+1 日
    fetch_end = None
    if simulate_date:
        fetch_end = (datetime.strptime(simulate_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    
    data = get_cleaned_data(ticker, "2022-01-01", end=fetch_end)
    if len(data) < 31: return

    # 載入帳戶狀態
    if os.path.exists(status_file):
        status = np.load(status_file, allow_pickle=True).item()
        balance, shares = status['balance'], status['shares']
        history = list(np.load(history_file))
    else:
        balance, shares, history = 100000.0, 0, []

    # 獲取狀態 (最後 30 天，不含最後一筆即今日收盤)
    window = data.iloc[-31:-1].values 
    mean, std = window.mean(axis=0), window.std(axis=0) + 1e-8
    state = ((window - mean) / std).flatten().astype(np.float32)

    # 預測
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0)
        action = model(s).argmax().item()

    # 今日結算 (使用最後一筆數據 data.iloc[-1])
    current_data = data.iloc[-1]
    price = current_data['Close'].item()
    fee, tax = 0.001425, 0.003

    if action == 1 and balance > price: # Buy
        can_buy = int(balance / (price * (1 + fee)))
        if can_buy > 0:
            balance -= can_buy * price * (1 + fee); shares += can_buy
    elif action == 2 and shares > 0: # Sell
        balance += shares * price * (1 - fee - tax); shares = 0

    net_worth = balance + (shares * price)
    history.append(net_worth)

    # 儲存
    np.save(status_file, {'balance': balance, 'shares': shares})
    np.save(history_file, np.array(history))
    
    # 繪圖 
    hist_np = np.array(history)
    if len(hist_np) > 1:
        returns = np.diff(hist_np) / (hist_np[:-1] + 1e-8)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        max_dd = (hist_np / np.maximum.accumulate(hist_np) - 1).min() * 100

        print(f"[{date.today()}] 淨資產: {net_worth:.2f} | 夏普: {sharpe:.2f} | 回撤: {max_dd:.2f}%")

        plt.figure(figsize=(12, 6))
        plt.plot(hist_np, label='AI Strategy', color='royalblue')
        plt.title(f"AI Performance: {ticker} (Updated: {date.today()})")
        plt.savefig("performance_report.png") # 儲存圖片
        plt.close() # 關閉畫布避免記憶體洩漏
    return net_worth

if __name__ == "__main__":
    from datetime import datetime
    
    # 預先載入模型
    state_dim = 30 * 8
    global_model = QNet(state_dim, 3)
    global_model.load_state_dict(torch.load("stock_dqn_model.pth", map_location="cpu"))
    global_model.eval()

    # 模式選擇：True 為跑過去 100 天測試，False 為每日伺服器更新
    TEST_MODE = True

    if TEST_MODE:
        # 清除舊紀錄以重新開始測試
        for f in ["trade_history.npy", "account_status.npy"]:
            if os.path.exists(f): os.remove(f)
            
        for n in range(1000, -1, -1):
            target_dt = (date.today() - timedelta(days=n)).strftime('%Y-%m-%d')
            # 排除週末 (yfinance 沒數據會報錯)
            if datetime.strptime(target_dt, '%Y-%m-%d').weekday() < 5:
                print(f"正在模擬: {target_dt}")
                run_daily_simulation(global_model, simulate_date=target_dt)
    else:
        # 正式伺服器每日執行
        run_daily_simulation(global_model)

