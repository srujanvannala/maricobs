import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Q-learning Trading Environment
# -----------------------------
class TradingEnv:
    def __init__(self, prices):
        self.prices = prices
        self.n_steps = len(prices)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.holding = False
        self.cash = 1000  # initial money
        self.shares = 0
        return (self.current_step, self.holding)

    def step(self, action):
        """
        Actions:
        0 = Hold
        1 = Buy
        2 = Sell
        """
        reward = 0
        price = self.prices[self.current_step]

        # Buy
        if action == 1 and not self.holding:
            self.shares = self.cash / price
            self.cash = 0
            self.holding = True

        # Sell
        elif action == 2 and self.holding:
            self.cash = self.shares * price
            self.shares = 0
            self.holding = False
            reward = self.cash - 1000  # profit/loss since start

        # Move to next step
        self.current_step += 1
        done = self.current_step == self.n_steps - 1

        state = (self.current_step, self.holding)
        return state, reward, done


# -----------------------------
# Q-learning Agent
# -----------------------------
def q_learning(prices, episodes=50, alpha=0.1, gamma=0.95, epsilon=0.1):
    env = TradingEnv(prices)
    n_states = len(prices)
    q_table = np.zeros((n_states, 2, 3))  # (time, holding, actions)

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            step, holding = state
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1, 2])  # exploration
            else:
                action = np.argmax(q_table[step, int(holding), :])  # exploitation

            next_state, reward, done = env.step(action)
            next_step, next_holding = next_state

            best_next_action = np.argmax(q_table[next_step, int(next_holding), :])
            q_table[step, int(holding), action] += alpha * (
                reward + gamma * q_table[next_step, int(next_holding), best_next_action]
                - q_table[step, int(holding), action]
            )
            state = next_state
    return q_table


# -----------------------------
# Streamlit App
# -----------------------------
st.title("📈 Q-learning Trading Demo (Marico Stock)")

# --- Upload CSV or generate random data ---
uploaded_file = st.file_uploader("Upload your Marico CSV (columns: Day, price)", type="csv")

if uploaded_file:
    df_data = pd.read_csv(uploaded_file)
    prices = df_data["price"].values   # <-- use your 'price' column
else:
    st.warning("No file uploaded. Using synthetic random prices instead.")
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100)) + 50
    prices = np.maximum(prices, 1)  # keep positive
    df_data = pd.DataFrame({"Day": np.arange(1, 101), "price": prices})

# Train Q-learning agent
q_table = q_learning(prices, episodes=200)

# Simulate trading
env = TradingEnv(prices)
state = env.reset()
done = False
actions = []
while not done:
    step, holding = state
    action = np.argmax(q_table[step, int(holding), :])
    actions.append(action)
    state, _, done = env.step(action)

# --- Fix: align lengths ---
min_len = min(len(prices), len(actions))
df = pd.DataFrame({
    "price": prices[:min_len],
    "Action": actions[:min_len]
})

# Visualization
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["price"], label="Stock Price", color="blue")

buy_signals = df[df["Action"] == 1]
sell_signals = df[df["Action"] == 2]

ax.scatter(buy_signals.index, buy_signals["price"], marker="^", color="green", label="Buy", s=100)
ax.scatter(sell_signals.index, sell_signals["price"], marker="v", color="red", label="Sell", s=100)

ax.set_title("Q-learning Trading Decisions (Marico Stock)")
ax.legend()
st.pyplot(fig)

st.write("✅ Green = Buy | ❌ Red = Sell | ➖ Nothing = Hold")
