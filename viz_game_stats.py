import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Load the cleaned CSV
# -----------------------------
df = pd.read_csv("data/lebron_core_games.csv")

# Parse dates and ensure proper sorting
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values("GAME_DATE").reset_index(drop=True)

# -----------------------------
# 2️⃣ Prepare Win/Loss Data
# -----------------------------
# Keep only valid W/L rows
df = df[df["WL"].isin(["W", "L"])].copy()
df["WIN"] = df["WL"].map({"W": 1, "L": 0})

# # -----------------------------
# # 3️⃣ Scatter Plot: Win vs Loss
# # -----------------------------
# plt.figure(figsize=(14, 4))
# plt.scatter(df["GAME_DATE"], df["WIN"], s=10, alpha=0.6, color="dodgerblue")
# plt.yticks([0, 1], ["Loss", "Win"])
# plt.title("LeBron James — Game Outcomes Over Time")
# plt.xlabel("Game Date")
# plt.ylabel("Result")
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()

# -----------------------------
# 4️⃣ Win Rate by Season
# -----------------------------
win_rate = df.groupby("season")["WIN"].mean()

plt.figure(figsize=(10, 4))
win_rate.plot(kind="bar", color="mediumseagreen")
plt.title("LeBron James — Win Rate by Season")
plt.ylabel("Win Percentage")
plt.xlabel("Season")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# -----------------------------
# 5️⃣ Rolling 20-Game Win % Trend
# -----------------------------
df["rolling_win_rate"] = df["WIN"].rolling(window=20, min_periods=5).mean()

plt.figure(figsize=(14, 4))
plt.plot(df["GAME_DATE"], df["rolling_win_rate"], color="darkorange")
plt.title("LeBron James — 20-Game Rolling Win Percentage")
plt.ylabel("Win % (last 20 games)")
plt.xlabel("Date")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

