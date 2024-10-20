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
model_loaded = False
camera = None
frame = None
capturing = False  # Renamed from capture_active
processing_active = False
frame = None
capture_images = []
groups = {}
ranking_data = []
camera_url = 'http://192.168.1.7/cam-hi.jpg'

def capture_frame():
    global camera, frame
    if camera is not None and camera.isOpened():
        ret, new_frame = camera.read()
        if ret:
            print("Frame captured successfully.")
            frame = new_frame
        else:
            print("Error: Could not read frame from camera.")
            frame = None
    else:
        # Se a câmera não está inicializada, não tente capturar o frame
        print("Error: Camera is not initialized or cannot be accessed.")
        frame = None





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

# Função contínua de captura de frames do feed de rede
def continuous_capture():
    global frame
    while True:
        capture_frame()
        time.sleep(0.1)  # Pequeno atraso para evitar sobrecarga



# Live verification
@app.route('/live_verification')
def live_verification():
    return render_template('live_verification.html', processing_active=processing_active)



def initialize_camera():
    global camera
    try:
        # Tentando inicializar a câmera via URL
        image_url = 'http://192.168.1.7/cam-hi.jpg'
        img_resp = urllib.request.urlopen(image_url)
        if img_resp.getcode() == 200:
            print("Camera URL accessed successfully.")
            return True
        else:
            print("Error: Could not access camera URL.")
            return False
    except Exception as e:
        print(f"Error accessing camera URL: {e}")
        return False




# Start live processing
# Start live processing
@app.route('/start_live_processing')
def start_live_processing():
    global processing_active

    # Iniciar a verificação da câmera
    if not initialize_camera():
        return "Error initializing camera", 500

    processing_active = True
    threading.Thread(target=process_live_video, daemon=True).start()  # Inicia a thread para o processamento de vídeo
    return redirect(url_for('live_verification'))




def close_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
        print("Camera released successfully.")





# Stop live processing
@app.route('/stop_live_processing')
def stop_live_processing():
    global camera, processing_active
    if camera is not None:
        camera.release()
        camera = None
    processing_active = False
    return redirect(url_for('live_verification'))

# Video feed
@app.route('/live_feed')
def live_feed():
    global frame, processing_active, model_loaded

    print("live_feed route called.")

    # Esperar até que o modelo esteja carregado
    if not model_loaded:
        return 'Model not loaded', 503

    if not processing_active or frame is None:
        # Retorna status 204 se o processamento não está ativo ou não há frame disponível
        return '', 204

    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    else:
        print("Error encoding frame.")
        return '', 204





@app.route('/check_model_status')
def check_model_status():
    global model_loaded, processing_active
    if model_loaded and processing_active:
        return '', 200
    else:
        return '', 503








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
    global processing_active, frame, model

    # Garantir que o modelo seja carregado antes de iniciar o processamento de vídeo
    if model is None:
        load_model()

    if model is None:
        print("Model loading failed. Stopping live processing.")
        processing_active = False
        return

    # URL da câmera
    image_url = 'http://192.168.1.7/cam-hi.jpg'

    while processing_active:
        try:
            img_resp = urllib.request.urlopen(image_url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgnp, -1)
            if model is not None:
                results = model(img)
                frame = np.squeeze(results.render())
            else:
                frame = img
        except Exception as e:
            print(f"Error processing live video: {e}")
            frame = None
        time.sleep(0.1)






import threading

camera_lock = threading.Lock()  # Lock global para gerenciar o acesso à câmera

def start_capture(camera_url, reset_delay=5):
    global capturing
    capturing = True

    def reconnect_camera():
        """Função para reinicializar a câmera após falhas."""
        nonlocal cap
        with camera_lock:  # Bloqueia o acesso à câmera
            print("Liberando a câmera...")
            cap.release()  # Libera a câmera
            time.sleep(reset_delay)  # Aguarda antes de reconectar
            print(f"Tentando reconectar a câmera após {reset_delay} segundos de espera...")
            cap = cv2.VideoCapture(camera_url)  # Tenta reconectar a câmera
            if cap.isOpened():
                print("Câmera reconectada com sucesso.")
            else:
                print(f"Erro: Não foi possível reconectar à câmera com URL: {camera_url}")

    # Inicializa a câmera pela primeira vez
    with camera_lock:  # Bloqueia o acesso à câmera
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir a câmera com a URL fornecida: {camera_url}")
            capturing = False
            return

    while capturing:
        with camera_lock:  # Bloqueia o acesso à câmera durante a captura
            ret, frame = cap.read()

        if ret:
            try:
                # Gera o nome do arquivo e salva a imagem capturada
                filename = f"capture_{int(time.time() * 1000)}.jpg"
                filepath = os.path.join('static/captures', filename)
                cv2.imwrite(filepath, frame)
                print(f"Imagem capturada e salva: {filename}")
            except Exception as e:
                print(f"Erro ao salvar a imagem capturada: {str(e)}")
                capturing = False
                break
        else:
            # Se não for possível capturar a imagem, reinicia a câmera
            print("Falha ao capturar o frame. Reinicializando a câmera...")
            reconnect_camera()

    with camera_lock:  # Bloqueia o acesso à câmera durante a liberação
        cap.release()
    print("Captura encerrada.")













# Stop capture
# Stop capture
@app.route('/stop_capture')
def stop_capture():
    global capturing
    capturing = False
    print("Interrompendo captura...")
    # Retornando redirecionamento para o dashboard ou uma página apropriada
    flash('Captura de imagens interrompida.', 'success')
    return redirect(url_for('dashboard'))  # Corrigido


@app.route('/start_capture', methods=['POST'])
def start_camera_capture():
    # Use um URL de câmera padrão se o URL não for fornecido via formulário
    camera_url = request.form.get('camera_url', 'http://192.168.1.7/cam-hi.jpg')  # URL da câmera enviada via formulário ou padrão
    interval = int(request.form.get('interval', 2))  # Intervalo entre capturas, se fornecido
    capture_thread = threading.Thread(target=start_capture, args=(camera_url, interval))
    capture_thread.start()
    flash('Captura de imagens iniciada.', 'success')
    return redirect(url_for('dashboard'))



@app.route('/stop_capture', methods=['POST'])
def stop_camera_capture():
    stop_capture()
    flash('Captura de imagens interrompida.', 'success')
    return redirect(url_for('dashboard'))





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
@app.route('/process_images', methods=['POST'])
def process_and_rank_images():
    global capture_images
    global ranking_data  # Ranking global

    if capture_images:
        accuracies = []
        for item in capture_images:
            results = item['results']
            if results is not None and len(results.xyxy[0]) > 0:
                # Pega as acurácias das detecções
                confidences = results.xyxy[0][:, 4]
                max_confidence = float(confidences.max())
                accuracies.append({'image': item['image'], 'accuracy': max_confidence, 'results': results})

        # Ordena as imagens pela acurácia e seleciona as top 5
        top5 = sorted(accuracies, key=lambda x: x['accuracy'], reverse=True)[:5]
        avg_accuracy = sum([x['accuracy'] for x in top5]) / len(top5) if top5 else 0

        # Adiciona as melhores imagens ao ranking
        group_name = session.get('group_name', 'Anônimo')
        ranking_entry = {'group': group_name, 'accuracy': avg_accuracy, 'images': []}

        # Salva as imagens com resultados
        for idx, item in enumerate(top5):
            img_path = item['image']
            img = cv2.imread(img_path)
            results = item['results']
            img_with_results = np.squeeze(results.render())
            result_img_name = f'{group_name}_result_{idx}.jpg'
            result_img_path = f'static/captures/{result_img_name}'
            cv2.imwrite(result_img_path, img_with_results)
            ranking_entry['images'].append(f'captures/{result_img_name}')

        # Atualiza o ranking
        load_ranking_data()
        ranking_data = [entry for entry in ranking_data if entry['group'] != group_name]  # Remove entradas antigas do grupo
        ranking_data.append(ranking_entry)
        save_ranking_data()

        flash(f'Resultados processados. Média de acurácia: {avg_accuracy:.2f}', 'success')
    else:
        flash('Nenhuma imagem capturada para processar.', 'error')

    return redirect(url_for('group_ranking'))


















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
    global model, model_loaded
    model_path = 'models/best.pt'
    if os.path.exists(model_path):
        print("Loading model from", model_path)
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            model.conf = 0.6
            model_loaded = True  # Modelo carregado com sucesso
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            model = None
            model_loaded = False
    else:
        print("Model file not found at", model_path)
        model = None
        model_loaded = False


# Inicia a aplicação Flask
if __name__ == '__main__':
    # Verificar se os diretórios necessários existem
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('static/captures'):
        os.makedirs('static/captures')
    load_groups()
    app.run(host='0.0.0.0', port=5000, debug=True)