import pandas as pd
import numpy as np
import requests
import sys  

def procesar_datos(url, columnas, bins, labels, nombre_archivo):
    # Descargar los datos
    response = requests.get(url)
    if response.status_code == 200:
        with open('heart_failure_clinical_records_dataset.csv', 'w') as file:
            file.write(response.text)
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

    # Verificar valores faltantes
    if df.isnull().any().any():
        print("Hay valores faltantes en el DataFrame.")
    else:
        print("No hay valores faltantes en el DataFrame.")

    # Verificar filas duplicadas
    if df.duplicated().any():
        print("Hay filas duplicadas en el DataFrame.")
    else:
        print("No hay filas duplicadas en el DataFrame.")

    # Eliminar valores atípicos
    for columna in columnas:
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        filtro = (df[columna] >= Q1 - 1.5 * IQR) & (df[columna] <= Q3 + 1.5 *IQR)
        df = df.loc[filtro]

    # Categorizar edades
    df['categoria_edad'] = pd.cut(df['age'], bins=bins, labels=labels)

    # Guardar los datos
    df.to_csv(nombre_archivo, index = False)
    print(df)

if __name__ == '__main__':
    url = sys.argv[1] 

    columnas = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_sodium']
    bins = [0, 12, 19, 39, 59, np.inf]
    labels = ['Niño', 'Adolescente', 'Jóvenes adulto', 'Adulto', 'Adulto mayor']

    procesar_datos(url, columnas, bins, labels, 'Datos_procesados_PI6.csv')
