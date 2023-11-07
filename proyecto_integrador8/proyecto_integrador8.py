import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('proyecto_integrador8/datos_procesados.csv')

fig, axs = plt.subplots(1, 4, figsize=(10, 5))

# Anémicos

anemicos = df[df['anaemia'] == 1]['anaemia'].count()
anemicos_no = df[df['anaemia'] == 0]['anaemia'].count()

categoria_a = ['No', 'Sí']
valores_a = [anemicos_no, anemicos]

axs[0].pie(valores_a, labels=categoria_a, colors = ['lightcoral', 'lightskyblue'], startangle=90, autopct='%1.1f%%')
axs[0].set_title('Anémicos') 

# Diabéticos

diabeticos = df[df['diabetes'] == 1]['diabetes'].count()
diabeticos_no = df[df['diabetes'] == 0]['diabetes'].count()

categoria_d = ['No', 'Sí']
valores_d = [diabeticos_no, diabeticos]

axs[1].pie(valores_d, labels=categoria_d, colors = ['lightcoral', 'lightskyblue'], startangle=90, autopct='%1.1f%%')
axs[1].set_title('Diabéticos') 

# Fumadores

fumadores = df[df['smoking'] == 1]['smoking'].count()
fumadores_no = df[df['smoking'] == 0]['smoking'].count()

categoria_f = ['No', 'Sí']
valores_f = [fumadores_no, fumadores]

axs[2].pie(valores_f, labels=categoria_f, colors = ['lightcoral', 'lightskyblue'], startangle=90, autopct='%1.1f%%')
axs[2].set_title('Fumadores') 

# Muertos

muertos = df[df['DEATH_EVENT'] == 1]['DEATH_EVENT'].count()
muertos_no = df[df['DEATH_EVENT'] == 0]['DEATH_EVENT'].count()

categoria_m = ['No', 'Sí']
valores_m = [muertos_no, muertos]

axs[3].pie(valores_m, labels=categoria_m, colors = ['lightcoral', 'lightskyblue'], startangle=90, autopct='%1.1f%%')
axs[3].set_title('Muertos')


plt.tight_layout()
plt.show()
