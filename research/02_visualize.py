import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).parent
csv_path = (script_dir / ".." / "data" / "usdjpy_1h.csv").resolve()

df = pd.read_csv(csv_path, skiprows=3, header=None,
    names=["Datetime", "Close", "High", "Low", "Open", "Volume"])

df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"])

df["Datetime"] = df["Datetime"].astype(str).str.slice(0, 19)
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
df = df.dropna(subset=["Datetime"])
df = df.set_index("Datetime")

print(f"件数：{len(df)}")
print(df["Close"].head())

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(df.index, df["Close"], linewidth=0.8, color="blue")
ax.set_title("USDJPY 1H 2Years")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()