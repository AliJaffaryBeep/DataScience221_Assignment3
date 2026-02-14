import pandas as pd
import matplotlib.pyplot as plt

crime_data = pd.read_csv("data/crime1.csv")
x = pd.to_numeric(crime_data["ViolentCrimesPerPop"], errors="coerce").dropna()

plt.figure()
plt.hist(x, bins=30, edgecolor="black")
plt.title("Histogram of ViolentCrimesPerPop")
plt.xlabel("ViolentCrimesPerPop")
plt.ylabel("Frequency")
plt.show()


plt.figure()
plt.boxplot(x, vert=True)
plt.title("Box Plot of ViolentCrimesPerPop")
plt.xlabel("ViolentCrimesPerPop")
plt.ylabel("Value")
plt.show()

"""

The histogram may exhibit a small amount of skew, but it looks generally bell-shaped and somewhat regularly distributed.
Additionally, it indicates that the data is distributed moderately between 0 and 1, 
with the majority of the data seeming concentrated between approximately 0.2 and 0.6.
According to the box plot, half of the values fall below the median, which is approximately 0.4.
A small right skew in the data is suggested by the median, which is marginally below the box's center.
There aren't any obvious outliers because the box plot doesn't display any points outside of the whiskers.

"""