import pandas as pd
from datasets import load_dataset

dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

# Pregunta 1
df = pd.DataFrame(data)
print(df)

# Pregunta 2
df_personas_perecieron = df[df['is_dead'] == 1]
print(df_personas_perecieron)

df_personas_no_perecieron = df[df['is_dead'] == 0]
print(df_personas_no_perecieron)

# Pregunta 3
promedio_1 = df[df['is_dead'] == 1]['age'].mean()
print(promedio_1)

promedio_2 = df[df['is_dead'] == 0]['age'].mean()
print(promedio_2)
