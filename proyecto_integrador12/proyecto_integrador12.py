import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

# 1. Eliminación de columnas

df = pd.read_csv('proyecto_integrador12/Datos_procesados.csv')
df = df.drop('categoria_edad', axis=1)

# 2. Partición del dataset en conjunto de entrenamiento y test

# Definir las variables independientes (X) y la variable dependiente (y)
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Random forest

# Crear el clasificador de Random Forest
clf = RandomForestClassifier(n_estimators=10, random_state=42)

# Ajustar el clasificador al conjunto de entrenamiento
clf.fit(X_train, y_train)

# Predecir las clases para el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)

# 4. matriz de confusión

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Crear una figura
fig, ax = plt.subplots()

# Crear un mapa de calor usando matshow
cax = ax.matshow(cm, cmap=plt.cm.Blues)

# Mostrar las etiquetas de las clases
plt.title('Matriz de confusión')
plt.xlabel('Clase predicha')
plt.ylabel('Clase real')

# Mostrar las barras de colores
fig.colorbar(cax)

# Mostrar los valores en las celdas
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), va='center', ha='center')

plt.show()

# 5. Calcular F1-Score y comparar con el accuracy 

# Calcular el F1-Score
f1 = f1_score(y_test, y_pred)

print(f'El F1-Score del Random Forest en el conjunto de prueba es: {f1}')
print(f'La precisión del Random Forest en el conjunto de prueba es: {accuracy}')

# 6. Cambiar los valores de los parámetros del random forest

# Definir los parámetros para la búsqueda de cuadrícula
params = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
}

# Crear el clasificador de Random Forest
clf = RandomForestClassifier(random_state=42)

# Crear la búsqueda de cuadrícula
grid_search = GridSearchCV(clf, params, cv=5, scoring='f1')

# Ajustar la búsqueda de cuadrícula al conjunto de entrenamiento
grid_search.fit(X_train, y_train)

# Obtener el mejor estimador
best_clf = grid_search.best_estimator_

# Predecir las clases para el conjunto de prueba
y_pred = best_clf.predict(X_test)

# Calcular el F1-Score y la precisión
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f'El F1-Score del Random Forest en el conjunto de prueba es: {f1}')
print(f'La precisión del Random Forest en el conjunto de prueba es: {accuracy}')
