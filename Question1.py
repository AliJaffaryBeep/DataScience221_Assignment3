import pandas as pd

crime_data = pd.read_csv("data/crime1.csv")
x = pd.to_numeric(crime_data["ViolentCrimesPerPop"], errors="coerce").dropna()

summary_stats = x.agg(["mean", "median", "std", "min", "max"]).to_frame(name="ViolentCrimesPerPop")

print(summary_stats)

''' 
Mean vs median:
If the mean is bigger than the median, it's usually right-skewed (a few high values pull the mean up).
If the mean is smaller than the median, it's usually left-skewed.
Outliers:
The mean changes more when there are extreme values because it uses every value in the average.
The median is steadier since it's based on the middle value, so a few extreme points usually don't move it much.

'''