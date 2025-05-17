
# Primero importo las cosas que necesito
import pandas as pd  # Para manejar los datos
from sklearn.model_selection import train_test_split  # Para dividir los datos
from sklearn.preprocessing import StandardScaler  # Para normalizar
import tensorflow as tf  # Para la red neuronal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt  # Para gráficos

# 1. Cargar los datos
print("Cargando datos...")
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
nombres_columnas = ['MPG', 'Cilindros', 'Desplazamiento', 'Caballos', 'Peso', 
                   'Aceleracion', 'Año', 'Origen']

# Leer el archivo (tuve que buscar cómo se hacía)
datos = pd.read_csv(url, names=nombres_columnas, na_values='?', 
                    comment='\t', sep=' ', skipinitialspace=True)

# 2. Limpiar los datos
print("Limpiando datos...")
datos = datos.dropna()  # Quitar filas con datos faltantes

# Cambiar el origen a números (1=USA, 2=Europe, 3=Japan)
datos['Origen'] = datos['Origen'].replace({1: 'USA', 2: 'Europe', 3: 'Japan'})

# Convertir a variables dummy (esto lo copié de internet)
datos = pd.get_dummies(datos, prefix='', prefix_sep='')

# 3. Preparar los datos
X = datos.drop('MPG', axis=1)  # Todas las columnas menos MPG
y = datos['MPG']  # Solo MPG

# Dividir en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos (me dijeron que es importante)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Crear el modelo de red neuronal
print("Creando modelo...")
modelo = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Capa entrada
    Dense(64, activation='relu'),  # Capa oculta
    Dense(1)  # Capa salida (1 valor: MPG)
])

# Configurar el modelo (esto costó entenderlo)
modelo.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae', 'mse'])

# 5. Entrenar el modelo
print("Entrenando modelo...")
historia = modelo.fit(X_train, y_train, 
                     epochs=1000, 
                     validation_split=0.2, 
                     verbose=1,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

# 6. Evaluar el modelo
print("\nEvaluando modelo...")
perdida, mae, mse = modelo.evaluate(X_test, y_test, verbose=0)
print(f'Error absoluto medio en prueba: {mae:.2f} MPG')

# 7. Hacer gráficos (copié esto de un ejemplo)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(historia.history['mae'], label='Entrenamiento')
plt.plot(historia.history['val_mae'], label='Validación')
plt.title('Error Absoluto Medio')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historia.history['loss'], label='Entrenamiento')
plt.plot(historia.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.legend()

plt.tight_layout()
plt.show()

# 8. Hacer algunas predicciones
print("\nHaciendo predicciones...")
ejemplos = X_test[:5]
predicciones = modelo.predict(ejemplos).flatten()

print("\nComparación predicciones vs valores reales:")
for i in range(5):
    print(f"Predije: {predicciones[i]:.1f} MPG | Real: {y_test.iloc[i]:.1f} MPG")
