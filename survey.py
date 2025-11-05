import pandas as pd
import random

people = ["Santiago","Henry","Olivia",
    "Sarah","Jack","Andrew",
    "Mac","Kyan Fogarty","Jordan Terrell Carter",
    "Steven","Kevin","Alice","Bob","Eve","Charlie","Mallory","Trent","Peggy","Victor","Walter","Yvonne","Zara","Quinn","Riley","Sam"
]


Devices = ["Smartphone",
    "Smart TV","Laptop/Desktop"
    "Tablet","Other"
]

Streaming_apps = ["Movies","TV" 
    "shows/series",
    "Sports","News",
    "Other"
]

satisfied = [random.randint(1,10)]

average_time = [6,
    3,
    0,
    2,
    10
]

amount_of_subs = [0,1,2,3,6]


data = pd.DataFrame({
    'Names: ':people,
    'Devices: ': [random.choice(Devices) for _ in people],
    'Streaming services: ': [random.choice(Streaming_apps) for _ in people],
    'Average time: ': [random.choice(average_time) for _ in people],
    'Satisfaction: ': [random.choice(satisfied)  for _ in people]
    })

efData = pd.DataFrame(data)
#efData.to_csv("ef.csv", index=False)
print(efData.describe())