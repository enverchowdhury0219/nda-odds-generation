import pandas as pd
import matplotlib.pyplot as plt

# adding this comment from cursor to see if if works!

df = pd.read_csv("data/lebron_core_games.csv")

# Parse dates and ensure proper sorting
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values("GAME_DATE").reset_index(drop=True)

# Keep only valid W/L rows
df = df[df["WL"].isin(["W", "L"])].copy()
df["WIN"] = df["WL"].map({"W": 1, "L": 0})


# Win Rate by Season: Shows the percentage of games LeBron’s teams won
# each season, highlighting how his success rate changes year to year.
win_rate = df.groupby("season")["WIN"].mean()

plt.figure(figsize=(10, 4))
win_rate.plot(kind="bar", color="mediumseagreen")
plt.title("LeBron James — Win Rate by Season")
plt.ylabel("Win Percentage")
plt.xlabel("Season")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# Rolling 20-Game Win % Trend:Plots the moving average of wins across
# the last 20 games to reveal short-term streaks and performance momentum 
# over time.
df["rolling_win_rate"] = df["WIN"].rolling(window=20, min_periods=5).mean()

plt.figure(figsize=(14, 4))
plt.plot(df["GAME_DATE"], df["rolling_win_rate"], color="blue")
plt.title("LeBron James — 20-Game Rolling Win Percentage")
plt.ylabel("Win % (last 20 games)")
plt.xlabel("Date")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# raw PTS per game
plt.figure(figsize=(14, 5))
plt.plot(df["GAME_DATE"], df["PTS"], color="red", linewidth=0.8, label="Game Points")
plt.title("LeBron James — Points per Game (All Seasons)")
plt.xlabel("Game Date")
plt.ylabel("Points Scored")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()