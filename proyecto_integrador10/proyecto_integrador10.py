import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 1. Eliminación de columnas

df = pd.read_csv('proyecto_integrador10/Datos_procesados.csv')
X = df.drop(['DEATH_EVENT','age', 'categoria_edad'], axis=1)

#2. Regresión lineal

# Usar la columna 'age' como vector objetivo
y = df['age']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear una instancia del modelo
model = LinearRegression()

# Ajustar el modelo a los datos de entrenamiento
model.fit(X_train, y_train)

# 3. Hacer predicciones en los datos de prueba

y_pred = model.predict(X_test)

# Comparar las predicciones con los valores reales
comparacion = pd.DataFrame({'Edad real': y_test, 'Edad predicha': y_pred})

print(comparacion)

# 4. Error cuadrático medio.

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)

print('Error cuadrático medio:', mse)
