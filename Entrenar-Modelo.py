#Entrenar el modelo
print("\nEntrenando modelo..."
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
