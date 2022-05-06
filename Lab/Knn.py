import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing

print("M.NAVEEN 19BCN7185")

df = pd.read_csv("Iris.csv")
print(f"COLUMNS : \n{df.columns}")
print(f"DATASET \n{df}")
sns.pairplot(df, hue='Species')
plt.show()

x = df.drop(columns=['Id', 'Species'])
label_encoder = preprocessing.LabelEncoder()
df["Species"] = label_encoder.fit_transform(df['Species'])
y = df['Species']

print(f"TRAIN DATASET \n{x}\nTEST DATASET {y}")
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x.values, y)
flower = neigh.predict([[4.4, 2.9, 1.4, 0.2]])
if flower == 0:
    print("Specie Is : SETOSA")
elif flower == 1:
    print("Specie Is : VERSICOLOR")
else:
    print("Specie Is : VIRGINICA")
