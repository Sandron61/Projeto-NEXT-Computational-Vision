# app.py

from flask import Flask, render_template, request, redirect, url_for, session, Response, flash, make_response
import os
import cv2
import torch
import numpy as np
import urllib.request
import threading
import time
import json
import pathlib
import traceback
# Fix for Windows systems where PosixPath is not available
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key

# Global variables
model = None
capturing = False  # Renamed from capture_active
processing_active = False
frame = None
capture_images = []
groups = {}
ranking_data = []


# Load ranking data from file
def load_ranking_data():
    global ranking_data
    if os.path.exists('ranking.json'):
        with open('ranking.json', 'r') as f:
            ranking_data = json.load(f)
    else:
        ranking_data = []

# Save ranking data to file
def save_ranking_data():
    global ranking_data
    with open('ranking.json', 'w') as f:
        json.dump(ranking_data, f)

# Load groups data from file
def load_groups():
    global groups
    if os.path.exists('groups.json'):
        with open('groups.json', 'r') as f:
            groups = json.load(f)
    else:
        groups = {}

# Save groups data to file
def save_groups():
    global groups
    with open('groups.json', 'w') as f:
        json.dump(groups, f)

# Index route (login)
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        # Simple authentication example
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = 'Usuário ou senha inválidos'
    return render_template('login.html', error=error)

# Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')



# Live verification
@app.route('/live_verification')
def live_verification():
    global processing_active
    return render_template('live_verification.html', processing_active=processing_active)


# Start live processing
@app.route('/start_live_processing')
def start_live_processing():
    global processing_active
    print("start_live_processing route called.")
    if not processing_active:
        processing_active = True
        print("Starting live processing...")
        threading.Thread(target=process_live_video).start()
        flash('Processamento ao vivo iniciado', 'success')
    return redirect(url_for('live_verification'))




# Stop live processing
@app.route('/stop_live_processing')
def stop_live_processing():
    global processing_active
    if processing_active:
        processing_active = False
        print("Stopping live processing...")
        flash('Processamento ao vivo parado', 'info')
    return redirect(url_for('live_verification'))

# Video feed
@app.route('/live_feed')
def live_feed():
    global frame
    print("live_feed route called.")
    if frame is not None:
        ret, buffer = cv2.imencode('.jpg', frame)
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    else:
        print("Frame is None in live_feed.")
        return '', 204



def gen_frames():
    global frame
    while True:
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

def process_live_video():
    global frame
    global processing_active
    global model

    # Load model if not loaded
    if model is None:
        load_model()

    # Replace with your image URL
    image_url = 'http://192.168.1.7/cam-hi.jpg'  # Update this URL

    while processing_active:
        try:
            img_resp = urllib.request.urlopen(image_url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgnp, -1)
            results = model(img)
            frame = np.squeeze(results.render())
        except Exception as e:
            print(f"Error processing live video: {e}")
            frame = None
        time.sleep(0.1)




# Start capture
@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capturing
    capturing = True
    print("Capture started.")
    flash('Captura iniciada', 'success')
    # Start the capture_images_func in a new thread
    threading.Thread(target=capture_images_func).start()
    return redirect(url_for('live_verification'))


# Stop capture
@app.route('/stop_capture')
def stop_capture():
    global capturing
    capturing = False
    print("Capture stopped.")
    flash('Captura parada', 'info')
    # Add any additional logic for stopping capture
    return redirect(url_for('live_verification'))

# Capture images function
def capture_images_func():
    global capturing
    global capture_images
    capture_images = []
    count = 0

    # Load model if not loaded
    if model is None:
        load_model()

    # Replace with your image URL
    image_url = 'http://192.168.1.7/cam-hi.jpg'  # Update this URL

    while capturing and count < 50:
        try:
            img_resp = urllib.request.urlopen(image_url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)
            results = model(img)
            # Save image and results
            timestamp = int(time.time() * 1000)
            img_name = f'static/captures/capture_{timestamp}.jpg'
            cv2.imwrite(img_name, img)
            capture_images.append({'image': img_name, 'results': results})
            count += 1
        except Exception as e:
            print(f"Error capturing images: {e}")
        time.sleep(0.5)


# Process results
@app.route('/process_results')
def process_results():
    global capture_images
    global ranking_data  # Declare ranking_data as global
    if capture_images:
        # Ensure directory exists
        if not os.path.exists('static/captures'):
            os.makedirs('static/captures')

        # Process captured images, select top 5 based on accuracy
        accuracies = []
        for item in capture_images:
            # Assuming results.xyxy[0] contains detections
            results = item['results']
            if results is not None and len(results.xyxy[0]) > 0:
                # Confidence scores are in results.xyxy[0][:, 4]
                confidences = results.xyxy[0][:, 4]
                max_confidence = float(confidences.max())
                accuracies.append({'image': item['image'], 'accuracy': max_confidence, 'results': results})
        # Sort by accuracy
        top5 = sorted(accuracies, key=lambda x: x['accuracy'], reverse=True)[:5]
        avg_accuracy = sum([x['accuracy'] for x in top5]) / 5 if top5 else 0
        # Update ranking
        group_name = session.get('group_name', 'Anônimo')
        ranking_entry = {'group': group_name, 'accuracy': avg_accuracy, 'images': []}

        # Save the top images with rendered results
        for idx, item in enumerate(top5):
            img_path = item['image']
            img = cv2.imread(img_path)
            results = item['results']
            img_with_results = np.squeeze(results.render())
            # Save the image with results
            result_img_name = f'{group_name}_result_{idx}.jpg'
            result_img_path = f'static/captures/{result_img_name}'
            cv2.imwrite(result_img_path, img_with_results)
            # Add to ranking entry
            ranking_entry['images'].append(f'captures/{result_img_name}')

        # Save ranking data
        load_ranking_data()
        # Remove previous entry of the group if exists
        ranking_data = [entry for entry in ranking_data if entry['group'] != group_name]
        ranking_data.append(ranking_entry)
        save_ranking_data()
        flash(f'Resultados processados. Média de acurácia: {avg_accuracy:.2f}', 'success')
    else:
        flash('Nenhuma imagem capturada para processar.', 'error')
    return redirect(url_for('live_verification'))

# View processed images
@app.route('/view_processed_images')
def view_processed_images():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    group_name = session.get('group_name', 'Anônimo')
    load_ranking_data()
    group_entry = next((entry for entry in ranking_data if entry['group'] == group_name), None)
    if group_entry and 'images' in group_entry:
        images = group_entry['images']
    else:
        images = []
    return render_template('view_processed_images.html', images=images)

# Register group
@app.route('/register_group', methods=['GET', 'POST'])
def register_group():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        group_name = request.form['group_name']
        session['group_name'] = group_name
        groups[group_name] = {'model': None}
        save_groups()
        flash(f'Grupo {group_name} registrado com sucesso.', 'success')
        return redirect(url_for('dashboard'))
    return render_template('register_group.html')

# Group ranking
@app.route('/group_ranking')
def group_ranking():
    load_ranking_data()
    sorted_ranking = sorted(ranking_data, key=lambda x: x['accuracy'], reverse=True)
    return render_template('group_ranking.html', ranking_data=sorted_ranking)

# Upload model
@app.route('/upload_model', methods=['GET', 'POST'])
def upload_model():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    current_model = None
    if request.method == 'POST':
        if 'model_file' in request.files:
            model_file = request.files['model_file']
            if model_file.filename != '':
                model_path = os.path.join('models', 'best.pt')
                model_file.save(model_path)
                flash('Modelo enviado com sucesso.', 'success')
                load_model()
            else:
                flash('Nenhum arquivo selecionado.', 'error')
        else:
            flash('Nenhum arquivo enviado.', 'error')
    if os.path.exists('models/best.pt'):
        current_model = 'best.pt'
    return render_template('upload_model.html', current_model=current_model)

# View results (if needed)
@app.route('/view_results')
def view_results():
    load_ranking_data()
    sorted_ranking = sorted(ranking_data, key=lambda x: x['accuracy'], reverse=True)
    return render_template('view_results.html', ranking_data=sorted_ranking)

# Load model
def load_model():
    global model
    model_path = 'models/best.pt'  # Update this path if necessary
    if os.path.exists(model_path):
        print("Loading model from", model_path)
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            model.conf = 0.6
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            model = None
    else:
        print("Model file not found at", model_path)
        model = None


if __name__ == '__main__':
    # Create necessary directories if they don't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('static/captures'):
        os.makedirs('static/captures')
    load_groups()
    app.run(host='0.0.0.0', port=5000, debug=True)
