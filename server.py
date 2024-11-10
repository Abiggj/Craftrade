from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import tensorflow.keras.backend as backend
import tensorflow.keras.models as models
import tensorflow.keras.preprocessing as preprocessing
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

generator = models.load_model('.\model_files\generator_model.h5', custom_objects={'wasserstein_loss': wasserstein_loss})
discriminator = models.load_model('./model_files/discriminator_model.h5', custom_objects={'wasserstein_loss': wasserstein_loss})

with open('./model_files/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

with open('./model_files/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text_input = data.get('text_input')
    last_real_ohlcv = data.get('last_real_ohlcv')

    if not text_input or last_real_ohlcv is None:
        return jsonify({'error': "Invalid input"}), 400

    last_real_ohlcv = scaler.transform([last_real_ohlcv])
    text_input = tokenizer.texts_to_sequences([text_input])
    text_input = preprocessing.sequence.pad_sequences(text_input, maxlen=100)

    generated_ohlcv = generator.predict([text_input, last_real_ohlcv])
    generated_ohlcv = scaler.inverse_transform(generated_ohlcv)

    return jsonify({'generated_ohlcv': generated_ohlcv.flatten().tolist()})


@app.route('/last_entry', methods=['GET'])
def get_last_entry():
    # Load the CSV file
    data = pd.read_csv('./model_files/data.csv')

    # Identify columns between 'date' and 'news'
    date_index = data.columns.get_loc("Date")
    news_index = data.columns.get_loc("news")
    ohlcv_columns = data.columns[date_index + 1: news_index]  # Columns between 'date' and 'news'

    # Get the last row's values for the selected columns
    last_row = data.iloc[-1]
    last_ohlcv = last_row[ohlcv_columns].tolist()
    last_news = last_row['news']  # Assuming 'news' is the column name

    # Send column names instead of dates for the x-axis
    column_names = ohlcv_columns.tolist()

    return jsonify({
        'last_ohlcv': last_ohlcv,
        'last_news': last_news,
        'column_names': column_names
    })


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Server is up and running!!"})

if __name__ == '__main__':
    app.run(debug=True)
