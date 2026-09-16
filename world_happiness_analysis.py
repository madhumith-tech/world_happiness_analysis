import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("2019.csv")

df = df[["Score", "GDP per capita", "Social support",
         "Healthy life expectancy", "Freedom to make life choices"]]

corr = df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, mask=mask)

plt.title("World Happiness Correlation")

plt.show()

strongest = corr["Score"].drop("Score").abs().idxmax()

print("Strongest:", strongest)

print("Correlation:", corr.loc["Score", strongest])

sns.regplot(x=df[strongest], y=df["Score"])

plt.title("Strongest Relationship")

plt.xlabel(strongest)

plt.ylabel("Happiness Score")

plt.show()