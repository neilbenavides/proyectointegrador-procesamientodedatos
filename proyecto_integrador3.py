import pandas as pd
from datasets import load_dataset

dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

df = pd.DataFrame(data)

# Pregunta 1
print(df.dtypes)

# Pregunta 2
hombres_fuman = df[(df['is_male'] == True) & (df['is_smoker'] == True)]['is_male'].count()
mujeres_fuman = df[(df['is_male'] == False) & (df['is_smoker'] == True)]['is_male'].count()

print('Cantidad de hombres fumadores:', hombres_fuman)
print('Cantidad de mujeres fumadoras:', mujeres_fuman)
