import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

df = pd.read_csv("ef2.csv")
ef = pd.DataFrame(df)

print("-_"*20)
print("Head of the data frame")
print(ef.head())

print("-_"*20)
print("Tail of the data frame")
print(ef.tail())

print("-_"*20)
print("Summary of the data frame")
print(ef.info())

print("-_"*20)
print("Statisitical Analysis")
print(round(ef.describe()))

print("-_"*20)
print("People with an Average Time Greater than 5") 
print(ef[ef['Average time: ']>5])

print("-_"*20)
print("Count of People in each device")
print(ef["Devices: "].value_counts())

print("-_"*20)
print("Count of People in each streaming service")
print(ef["Streaming services: "].value_counts())


bar_chart = ef["Devices: "].value_counts().plot(kind="bar", color=["blue", "orange", "green", "red", "purple"])
plt.title("Devices Used by People")
plt.xlabel("Device")
plt.ylabel("Number of People")
plt.show()

#The autopct = "%1.1f%" is used to calucate the percentages of of each pie slice.
pie_chart = ef["Streaming services: "].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Streaming Services Used by People")
plt.ylabel("")
plt.show()

#I learned that sns can be used to visualize statistical graphics so I learned some ways to use it.
scatter_plot = sns.scatterplot(data=ef, x="Average time: ", y="Satisfaction: ", hue="Devices: ")
plt.title("Average Time vs Satisfaction")
plt.xlabel("Average Time (hours)")
plt.ylabel("Satisfaction (1-10)")
plt.show()

satisfaction_boxplot = sns.boxplot(x="Devices: ", y="Satisfaction: ", data=ef)
plt.title("Satisfaction by Device")
plt.xlabel("Device")
plt.ylabel("Satisfaction (1-10)")
plt.show()

average_time_boxplot = sns.boxplot(x="Devices: ", y="Average time: ", data=ef)
plt.title("Average Time by Device")
plt.xlabel("Device")
plt.ylabel("Average Time (hours)")
plt.show()

average_time_piechart = ef["Average time: "].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Average Time Distribution")
plt.ylabel("")
plt.show()

stream_services_histogram = ef["Streaming services: "].value_counts().plot(kind="bar", color=["cyan", "magenta", "yellow", "black", "grey"])
plt.title("Streaming Services Distribution")
plt.xlabel("Streaming Service")
plt.ylabel("Number of People")
plt.show()