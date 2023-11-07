
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE



df = pd.read_csv('proyecto_integrador9/heart_failure_clinical_records_dataset.csv')
df_drop = df.drop('DEATH_EVENT', axis=1)

df_array = df_drop.values

y = df['DEATH_EVENT'].values
X = df_array

X_embedded = TSNE(
    n_components=3,
    learning_rate='auto',
    init='random',
    perplexity=3
).fit_transform(X)

df = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'z': X_embedded[:, 2], 'label': y})

# Creamos el gráfico de dispersión 3D
fig = go.Figure()

# Añadimos los puntos al gráfico
for label, color in zip([0, 1], ['blue', 'red']):
    df_filtered = df[df['label'] == label]
    fig.add_trace(go.Scatter3d(
        x=df_filtered['x'], y=df_filtered['y'], z=df_filtered['z'],
        mode='markers',
        marker=dict(color=color),
        name='Vivo' if label == 0 else 'Muerto'
    ))

fig.show()
