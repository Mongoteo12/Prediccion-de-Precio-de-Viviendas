import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 
# Cargar el dataset
url = 'Viviendas16.csv'
df = pd.read_csv(url)

# Separar las características y la variable objetivo
X = df.drop(['Precio'], axis=1)  # Características
y = df['Precio']  # Variable objetivo

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir el modelo de red neuronal
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=1500, batch_size=32, verbose=0)

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=1500, batch_size=32, verbose=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario enviado por el usuario
    metros2 = float(request.form['metros2'])
    estrato = int(request.form['estrato'])
    ascensor = int(request.form['ascensor'])
    calentador = int(request.form['calentador'])
    habitaciones = int(request.form['habitaciones'])
    garaje = int(request.form['garaje'])
    deposito = int(request.form['deposito'])
    banos = int(request.form['banos'])

    # Procesar los datos
    datos_usuario = pd.DataFrame([[metros2], [estrato], [ascensor], [calentador], [habitaciones], [garaje], [deposito], [banos]])

    # Realizar la predicción utilizando el modelo
    requisitos_usuario_scaled = scaler.transform(datos_usuario)
    precio_predicho = model.predict(requisitos_usuario_scaled)[0][0]

    # Devolver la respuesta al cliente
    return jsonify((precio_predicho))

if __name__ == '__main__':
    app.run(debug=True)
