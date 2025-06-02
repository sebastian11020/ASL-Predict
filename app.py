from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import random
import base64
from io import BytesIO
from PIL import Image
import plotly.express as px
from flask import send_file
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

app = Flask(__name__)

# Cargar el modelo y etiquetas
model = load_model('asl_model.h5')
model.predict(np.zeros((1, 96, 96, 3)))  # Inicializa el modelo para evitar errores de carga tardía
labels = sorted(os.listdir('asl_alphabet_train'))

# Nombre de la última capa convolucional para Grad-CAM
last_conv_layer_name = "conv2d_2"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    explanation = None
    preview_image = None
    probabilities = None

    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file:
            # Leer imagen
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            preview_image = encode_image_to_base64(image)

            # Preprocesar
            image_input = preprocess_image(image)

            # Predicción
            prediction = model.predict(image_input)[0]
            max_index = np.argmax(prediction)
            result = labels[max_index]
            confidence = round(float(prediction[max_index]) * 100, 2)

            probabilities = sorted(
                [(label, float(prob * 100)) for label, prob in zip(labels, prediction)],
                key=lambda x: x[1],
                reverse=True
            )

            # Grad-CAM
            heatmap = make_gradcam_heatmap(image_input, model, last_conv_layer_name)
            explanation_img = overlay_heatmap(heatmap, image)
            explanation = encode_image_to_base64(explanation_img)

    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        preview_image=preview_image,
        letters=labels,
        probabilities=probabilities,
        explanation_image=explanation
    )

@app.route('/descargar-imagen')
def descargar_imagen():
    folder = os.path.join('static', 'asl')
    imagenes = [img for img in os.listdir(folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not imagenes:
        return "No hay imágenes disponibles", 404

    imagen = random.choice(imagenes)
    ruta = os.path.join(folder, imagen)

    return send_file(ruta, as_attachment=True, download_name=imagen)


@app.route('/embeddings', methods=['GET', 'POST'])
def embeddings():
    num_letters = int(request.form.get('num_letters', 5))
    num_images = int(request.form.get('num_images', 10))

    X = []
    y = []
    thumbnails = []

    base_dir = 'asl_alphabet_train'
    selected_labels = random.sample(labels, min(num_letters, len(labels)))

    for label in selected_labels:
        path = os.path.join(base_dir, label)
        images = os.listdir(path)
        selected = random.sample(images, min(num_images, len(images)))
        for img_name in selected:
            img_path = os.path.join(path, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Miniatura en base64
            thumb = cv2.resize(image, (64, 64))  # mejor tamaño para tooltip
            _, buffer = cv2.imencode('.png', thumb)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            img_data_uri = f"data:image/png;base64,{img_b64}"
            thumbnails.append(img_data_uri)

            X.append(preprocess_image_for_embedding(image))
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Extraer embeddings
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
    embeddings = feature_extractor.predict(X)

    # Reducir a 2D
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    
    confidences = np.linalg.norm(embeddings, axis=1)
    confidences = 100 * (confidences - confidences.min()) / (confidences.max() - confidences.min())  # escala 0-100
    confidences = np.round(confidences, 2)

    # Construir DataFrame
    df = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'label': y,
        'confidence': confidences,
    })

    # Agregar custom_data aparte
    df['custom_label'] = y
    df['custom_image'] = thumbnails

    fig = px.scatter(
        df, x='x', y='y', color='label',
        custom_data=['label', 'confidence']
    )

    fig.update_traces(
        marker=dict(size=10, opacity=0.8),
        hovertemplate="""
        <b>Letra:</b> %{customdata[0]}<br>
        <b>Confianza:</b> %{customdata[1]:.2f}%<br>
        <b>x:</b> %{x:.2f}<br>
        <b>y:</b> %{y:.2f}<br>
        <extra></extra>
        """
    )

   
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template(
        'embeddings.html',
        plot_html=plot_html,
        selected_letters=num_letters,
        selected_images=num_images
    )


# ---- Utilidades ----

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    return np.expand_dims(image, axis=0)

def preprocess_image_for_embedding(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    return image

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
    return output

# ---- Ejecutar la app ----

if __name__ == '__main__':
    app.run(debug=True)
