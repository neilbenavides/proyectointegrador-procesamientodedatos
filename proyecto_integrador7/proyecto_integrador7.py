import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('proyecto_integrador7/datos_procesados.csv')
edades_organizadas = df['age'].sort_values()

# 1. Graficar la distribución de edades con un histograma.

# calcular el número de bins
num_bins = np.ceil(1 + np.log2(len(edades_organizadas))).astype(int)


plt.hist(edades_organizadas, bins=num_bins, edgecolor='black')
plt.title('Distribución de Edades')
plt.xlabel('Edades')
plt.ylabel('Frecuencia')
plt.xlim(35,100)
plt.ylim(0,60)
plt.show()

# 2. Graficar histogramas agrupado por hombre y mujer.

anemicos_h = df[(df['anaemia'] == 1) & (df['sex']== 1)]['anaemia'].count()
anemicos_m = df[(df['anaemia'] == 1) & (df['sex']== 0)]['anaemia'].count()

diabeticos_h = df[(df['diabetes'] == 1) & (df['sex']== 1)]['diabetes'].count()
diabeticos_m = df[(df['diabetes'] == 1) & (df['sex']== 0)]['diabetes'].count()

fumadores_h = df[(df['smoking'] == 1) & (df['sex']== 1)]['smoking'].count()
fumadores_m = df[(df['smoking'] == 1) & (df['sex']== 0)]['smoking'].count()

muertos_h = df[(df['DEATH_EVENT'] == 1) & (df['sex']== 1)]['DEATH_EVENT'].count()
muertos_m = df[(df['DEATH_EVENT'] == 1) & (df['sex']== 0)]['DEATH_EVENT'].count()

hombres = [anemicos_h, diabeticos_h, fumadores_h, muertos_h] 
mujeres = [anemicos_m, diabeticos_m, fumadores_m, muertos_m]

categorias = ['anémicos', 'diabéticos','fumadores', 'muertos']

x = np.arange(len(categorias))

plt.bar(x - 0.2, hombres, 0.4, label='Hombres')
plt.bar(x + 0.2, mujeres, 0.4, label='Mujeres')
plt.title('Histograma Agrupado por Sexo')
plt.xlabel('Categorías')
plt.ylabel('Cantidad')
plt.xticks(x, categorias)
plt.legend(loc='upper right')
plt.show()
