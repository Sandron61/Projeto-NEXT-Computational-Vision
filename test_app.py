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
capture_images = []
continuous_capturing = False
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
        try:
            with open('groups.json', 'r') as f:
                groups = json.load(f)
            print(f"Grupos carregados de groups.json: {groups}")
        except Exception as e:
            print(f"Erro ao carregar grupos de groups.json: {e}")
            traceback.print_exc()
            groups = {}
    else:
        groups = {}
        print("Arquivo groups.json não encontrado. Iniciando com dicionário de grupos vazio.")

# Save groups data to file
def save_groups():
    global groups
    try:
        with open('groups.json', 'w') as f:
            json.dump(groups, f)
        print("Grupos salvos com sucesso em groups.json")
    except Exception as e:
        print(f"Erro ao salvar grupos em groups.json: {e}")
        traceback.print_exc()

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

@app.route('/start_continuous_capture', methods=['POST'])
def start_continuous_capture():
    """
    Inicia a captura contínua de imagens.
    """
    global continuous_capturing
    continuous_capturing = True

    def capture_images_continuously():
        global frame
        while continuous_capturing:
            if frame is not None:
                filename = f"capture_{int(time.time() * 1000)}.jpg"
                filepath = os.path.join('static/captures', filename)
                cv2.imwrite(filepath, frame)
                print(f"Imagem contínua capturada e salva: {filename}")
            time.sleep(2)  # Intervalo entre capturas contínuas

    # Inicia uma nova thread para captura contínua
    capture_thread = threading.Thread(target=capture_images_continuously)
    capture_thread.start()

    return "Captura contínua iniciada", 200

@app.route('/stop_continuous_capture', methods=['POST'])
def stop_continuous_capture():
    """
    Para a captura contínua de imagens.
    """
    global continuous_capturing
    continuous_capturing = False
    return "Captura contínua parada", 200

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
@app.route('/start_live_processing')
def start_live_processing():
    global processing_active

    # Iniciar a verificação da câmera
    if not initialize_camera():
        return "Error initializing camera", 500

    processing_active = True

    # Get group name from session
    group_name = session.get('group_name', 'Anônimo')

    threading.Thread(target=process_live_video, args=(group_name,), daemon=True).start()  # Pass group_name
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

@app.route('/capture_live_image', methods=['POST'])
def capture_live_image():
    """
    Captura uma imagem atual da transmissão ao vivo e a salva.
    """
    global frame
    if frame is not None:
        filename = f"capture_{int(time.time() * 1000)}.jpg"
        filepath = os.path.join('static/captures', filename)
        cv2.imwrite(filepath, frame)
        return "Imagem capturada com sucesso.", 200
    else:
        return "Nenhuma imagem disponível para capturar.", 500

def process_live_video(group_name, capture_images=True, capture_interval=2):
    """
    Função para processar o vídeo ao vivo e capturar imagens opcionalmente.
    """
    global processing_active, frame, model, capturing
    last_capture_time = time.time()  # Controle de tempo para capturar as imagens

    # Garantir que o modelo esteja carregado
    load_group_model(group_name)

    if model is None:
        print("Model loading failed. Stopping live processing.")
        processing_active = False
        return

    # URL da câmera
    image_url = 'http://192.168.1.7/cam-hi.jpg'

    # Garantir que o diretório do grupo exista
    group_capture_dir = os.path.join('static', 'captures', group_name)
    os.makedirs(group_capture_dir, exist_ok=True)

    while processing_active:
        try:
            with camera_lock:  # Garante que o acesso à câmera seja exclusivo
                img_resp = urllib.request.urlopen(image_url)
                imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                img = cv2.imdecode(imgnp, -1)

                # Processa a imagem com o modelo, se disponível
                if model is not None:
                    results = model(img)
                    frame = np.squeeze(results.render())
                else:
                    frame = img

                # Captura a imagem se o tempo de intervalo foi atingido e a captura estiver ativa
                if capture_images and capturing:
                    current_time = time.time()
                    if current_time - last_capture_time >= capture_interval:
                        # Nome do arquivo
                        filename = f"capture_{int(current_time * 1000)}.jpg"
                        filepath = os.path.join(group_capture_dir, filename)
                        # Salvar a imagem
                        cv2.imwrite(filepath, frame)
                        print(f"Imagem capturada e salva: {filename}")
                        # Atualizar o ranking_data ou outra estrutura com 'filename', não 'filepath'
                        # Por exemplo:
                        # img_info = {'image_filename': filename, 'class': detected_class, 'confidence': confidence_score}
                        # ranking_data.append(img_info)
                        last_capture_time = current_time

        except Exception as e:
            print(f"Error processing live video: {e}")
            frame = None
        time.sleep(0.1)


@app.route('/start_live_capture', methods=['POST'])
def start_live_capture():
    """
    Inicia a transmissão ao vivo com captura de imagens.
    """
    global capturing, processing_active
    capturing = True
    processing_active = True
    capture_interval = int(request.form.get('capture_interval', 2))  # Intervalo entre capturas

    # Get group name from session
    group_name = session.get('group_name', 'Anônimo')

    capture_thread = threading.Thread(target=process_live_video, args=(group_name, True, capture_interval))
    capture_thread.start()
    flash('Transmissão ao vivo e captura de imagens iniciadas.', 'success')
    return redirect(url_for('dashboard'))

@app.route('/stop_live_capture', methods=['POST'])
def stop_live_capture():
    """
    Para a transmissão ao vivo e a captura de imagens.
    """
    global capturing, processing_active
    capturing = False
    processing_active = False
    flash('Captura de imagens interrompida.', 'success')
    return redirect(url_for('dashboard'))

import threading

camera_lock = threading.Lock()  # Lock global para gerenciar o acesso à câmera

# Stop capture
@app.route('/stop_capture')
def stop_capture():
    global capturing
    capturing = False
    print("Interrompendo captura...")
    # Retornando redirecionamento para o dashboard ou uma página apropriada
    flash('Captura de imagens interrompida.', 'success')
    return redirect(url_for('dashboard'))

@app.route('/start_capture', methods=['POST'])
def start_camera_capture():
    # Use um URL de câmera padrão se o URL não for fornecido via formulário
    camera_url = request.form.get('camera_url', 'http://192.168.1.7/cam-hi.jpg')  # URL da câmera enviada via formulário ou padrão
    interval = int(request.form.get('interval', 2))  # Intervalo entre capturas, se fornecido
    # Get group name from session
    group_name = session.get('group_name', 'Anônimo')
    capture_thread = threading.Thread(target=start_capture, args=(camera_url, interval, group_name))
    capture_thread.start()
    flash('Captura de imagens iniciada.', 'success')
    return redirect(url_for('dashboard'))

@app.route('/stop_capture', methods=['POST'])
def stop_camera_capture():
    stop_capture()
    flash('Captura de imagens interrompida.', 'success')
    return redirect(url_for('dashboard'))

# Capture images function
def capture_images_func(group_name):
    global capturing
    global capture_images
    capture_images = []
    count = 0

    # Carregar o modelo do grupo
    load_group_model(group_name)

    if model is None:
        print("Model loading failed. Stopping capture.")
        capturing = False
        return

    # URL da câmera
    image_url = 'http://192.168.1.7/cam-hi.jpg'  # Atualize este URL conforme necessário

    # Garantir que o diretório do grupo exista
    group_capture_dir = os.path.join('static', 'captures', group_name)
    os.makedirs(group_capture_dir, exist_ok=True)

    while capturing and count < 50:
        try:
            img_resp = urllib.request.urlopen(image_url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)
            results = model(img)
            # Salvar imagem e resultados
            timestamp = int(time.time() * 1000)
            img_name = f'capture_{timestamp}.jpg'
            img_path = os.path.join(group_capture_dir, img_name)
            cv2.imwrite(img_path, img)
            capture_images.append({'image': img_path, 'results': results})
            count += 1
        except Exception as e:
            print(f"Error capturing images: {e}")
        time.sleep(0.5)


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
    return render_template('view_processed_images.html', images=images, group_name=group_name)


# Register group
@app.route('/register_group', methods=['GET', 'POST'])
def register_group():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        group_name = request.form['group_name']
        
        if 'model_file' not in request.files or request.files['model_file'].filename == '':
            flash('Por favor, envie um arquivo de modelo.', 'error')
            return redirect(url_for('register_group'))
        
        model_file = request.files['model_file']
        
        # Criar diretório para o grupo
        group_dir = os.path.join('models', group_name)
        os.makedirs(group_dir, exist_ok=True)
        
        # Salvar o modelo no diretório do grupo
        model_path = os.path.join(group_dir, 'model.pt')
        model_file.save(model_path)
        
        # Verificar se o arquivo foi salvo
        if os.path.exists(model_path):
            print(f"Modelo salvo com sucesso em: {model_path}")
        else:
            print(f"Erro ao salvar o modelo para o grupo {group_name}")
        
        # Carregar grupos existentes
        load_groups()
        
        # Registrar o grupo
        groups[group_name] = {'model': model_path}
        save_groups()
        print(f"Grupo {group_name} adicionado ao dicionário e salvo em groups.json")
        
        # Registrar o grupo na sessão
        session['group_name'] = group_name
        session['model_name'] = os.path.basename(model_path)
        
        flash(f'Grupo {group_name} registrado com sucesso e modelo carregado.', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('register_group.html')

# Rota para seleção de grupos
@app.route('/select_group', methods=['GET', 'POST'])
def select_group():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    load_groups()  # Certifique-se de carregar os grupos disponíveis

    if request.method == 'POST':
        selected_group = request.form.get('group_name')
        if selected_group in groups:
            session['group_name'] = selected_group
            session['model_name'] = os.path.basename(groups[selected_group]['model'])
            flash(f'Grupo {selected_group} selecionado com sucesso.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Grupo selecionado não encontrado.', 'error')

    group_names = list(groups.keys())
    return render_template('select_group.html', group_names=group_names)

# Group ranking
@app.route('/group_ranking')
def group_ranking():
    load_ranking_data()
    sorted_ranking = sorted(ranking_data, key=lambda x: x['accuracy'], reverse=True)
    return render_template('group_ranking.html', ranking_data=sorted_ranking)

def save_ranking_data():
    global ranking_data
    with open('ranking.json', 'w') as f:
        json.dump(ranking_data, f)

# Carregar ranking do arquivo
def load_ranking_data():
    global ranking_data
    if os.path.exists('ranking.json'):
        with open('ranking.json', 'r') as f:
            ranking_data = json.load(f)
    else:
        ranking_data = []

def load_group_model(group_name):
    global model, model_loaded, groups

    # Carregar grupos existentes
    load_groups()
    
    # Verificar se o grupo tem um modelo associado
    if group_name in groups and 'model' in groups[group_name]:
        model_path = groups[group_name]['model']
        
        print(f"Tentando carregar o modelo do grupo {group_name} de {model_path}")
        
        if os.path.exists(model_path):
            print(f"Carregando modelo de {model_path}...")
            try:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
                model.conf = 0.6
                model_loaded = True
                print(f"Modelo do grupo {group_name} carregado com sucesso.")
            except Exception as e:
                print(f"Erro ao carregar o modelo do grupo {group_name}: {e}")
                traceback.print_exc()
                model = None
                model_loaded = False
        else:
            print(f"Modelo do grupo {group_name} não encontrado em {model_path}.")
            model = None
            model_loaded = False
    else:
        print(f"Grupo {group_name} não tem um modelo associado.")
        model = None
        model_loaded = False

# Inicia a aplicação Flask
if __name__ == '__main__':
    # Verificar se os diretórios necessários existem
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('static/captures'):
        os.makedirs('static/captures')

    # Carregar grupos existentes
    load_groups()

    app.run(host='0.0.0.0', port=5000, debug=True)
