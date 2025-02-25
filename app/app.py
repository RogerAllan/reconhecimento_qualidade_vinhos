from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
CORS(app)

# Carregar o modelo e o scaler
with open('C:/Users/roger/Desktop/projeto/model/modelo_vinho_ensemble.pkl', 'rb') as arquivo_modelo:
    modelo = pickle.load(arquivo_modelo)

with open('C:/Users/roger/Desktop/projeto/model/scaler.pkl', 'rb') as arquivo_scaler:
    scaler = pickle.load(arquivo_scaler)

@app.route('/predict', methods=['POST'])
def predict():
    dados = request.get_json(force=True)

    # Verificar se todas as features estão presentes
    campos_esperados = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                        'pH', 'sulphates', 'alcohol']
    
    if not all(campo in dados for campo in campos_esperados):
        return jsonify({'erro': 'Campos faltando na requisição'}), 400

    # Extrair os valores na ordem correta
    try:
        dados_input = np.array([[float(dados[campo]) for campo in campos_esperados]])
    except ValueError:
        return jsonify({'erro': 'Valores inválidos nos campos'}), 400

    # Escalonar os dados
    dados_scaled = scaler.transform(dados_input)

    # Fazer a previsão
    previsao = modelo.predict(dados_scaled)
    probabilidade = modelo.predict_proba(dados_scaled)

    # Preparar o resultado com previsão e probabilidades
    resultado = {
        'previsao': int(previsao[0]),
        'probabilidade_alta_qualidade': f"{probabilidade[0][1]:.2%}",
        'probabilidade_baixa_qualidade': f"{probabilidade[0][0]:.2%}"
    }
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)
