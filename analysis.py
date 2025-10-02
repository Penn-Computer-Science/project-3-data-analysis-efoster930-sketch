import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

df = pd.read_csv("ef.csv")
ef = pd.DataFrame(df)
print("-_"*20)

print("Statisitical Analysis")
print(round(ef.describe()))

