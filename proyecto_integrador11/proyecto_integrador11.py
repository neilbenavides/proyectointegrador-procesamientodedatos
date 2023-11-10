import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# 1. Eliminación de columnas

df = pd.read_csv('proyecto_integrador11/Datos_procesados.csv')
df = df.drop('categoria_edad', axis=1)

# 2. Graficar la distribución de clases

plt.figure(figsize=(8, 6))
df['DEATH_EVENT'].value_counts().plot(kind='bar')
plt.title('Distribución de clases para Muertes')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()

# 3. Partición del dataset en conjunto de entrenamiento y test

# Definir las variables independientes (X) y la variable dependiente (y)
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Ajusta un árbol de decisión y calcula el accuracy

# Ajustar el árbol de decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predecir las clases para el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)

#5. Validación cruzada

# Definir los parámetros para la búsqueda de cuadrícula
params = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
}

# Crear el árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Crear la búsqueda de cuadrícula
grid_search = GridSearchCV(clf, params, cv=5, scoring='accuracy')

# Ajustar la búsqueda de cuadrícula al conjunto de entrenamiento
grid_search.fit(X_train, y_train)

# Obtener el mejor estimador
best_clf = grid_search.best_estimator_

# Predecir las clases para el conjunto de prueba
y_pred = best_clf.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)

print(f'La precisión del árbol de decisión en el conjunto de prueba es: {accuracy}')
