modelo = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Capa entrada
    Dense(64, activation='relu'),  # Capa oculta
    Dense(1)  # Capa salida (1 valor: MPG)
])

# Configurar el modelo (esto costó entenderlo)
modelo.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae', 'mse'])

