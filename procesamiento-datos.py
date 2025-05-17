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
