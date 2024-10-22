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

# Variáveis globais
#model = None
#model_loaded = False
camera = None
#frame = None
#capturing = False  # Renamed from capture_active
#processing_active = False
capture_images = []
continuous_capturing = False
groups = {}
ranking_data = []
ranking_data_lock = threading.RLock()
group_processors = {}
group_processors_lock = threading.Lock()

camera_url = 'http://192.168.1.7/cam-hi.jpg'



def default_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)
    

class GroupProcessor:
    def __init__(self, group_name):
        self.group_name = group_name
        self.model = None
        self.model_loaded = False
        self.processing_active = False
        self.capturing = False
        self.frame = None
        self.capture_thread = None
        self.load_model()
        self.ranking_data_lock = threading.RLock()
        self.last_capture_time = time.time()

    def load_model(self):
        # Load the model for the group
        if self.group_name in groups and 'model' in groups[self.group_name]:
            model_path = groups[self.group_name]['model']
            if os.path.exists(model_path):
                try:
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
                    self.model.conf = 0.6
                    self.model_loaded = True
                    print(f"Model for group {self.group_name} loaded successfully.")
                except Exception as e:
                    print(f"Error loading model for group {self.group_name}: {e}")
                    traceback.print_exc()
                    self.model = None
                    self.model_loaded = False
            else:
                print(f"Model for group {self.group_name} not found at {model_path}.")
                self.model = None
                self.model_loaded = False
        else:
            print(f"Group {self.group_name} does not have an associated model.")
            self.model = None
            self.model_loaded = False



    def start_processing(self):
        try:
            if not self.processing_active:
                self.processing_active = True
                self.capturing = True
                self.capture_thread = threading.Thread(target=self.process_live_video, daemon=True)
                self.capture_thread.start()
        except Exception as e:
            print(f"Error starting processing for group {self.group_name}: {e}")
            traceback.print_exc()
            self.processing_active = False
            self.capturing = False


    def stop_processing(self):
        self.processing_active = False
        self.capturing = False
        if self.capture_thread is not None:
            self.capture_thread.join()
            self.capture_thread = None

    def process_live_video(self):
        # Similar to the previous process_live_video function, but using self variables
        last_capture_time = time.time()
        model = self.model

        if not self.model_loaded:
            print(f"Model for group {self.group_name} is not loaded.")
            return

        # URL da câmera
        image_url = 'http://192.168.1.7/cam-hi.jpg'

        # Ensure the group capture directory exists
        group_capture_dir = os.path.join('static', 'captures', self.group_name)
        os.makedirs(group_capture_dir, exist_ok=True)

        while self.processing_active:
            try:
                img_resp = urllib.request.urlopen(image_url)
                imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                img = cv2.imdecode(imgnp, -1)

                # Process the image with the model
                results = model(img)
                self.frame = np.squeeze(results.render())

                # Process detections
                # (Implement the same logic as before, but use self variables and methods)
                self.process_detections(results, img, group_capture_dir)

            except Exception as e:
                print(f"Error in live video processing for group {self.group_name}: {e}")
                traceback.print_exc()
            time.sleep(0.1)

    def process_detections(self, results, img, group_capture_dir):
        current_time = time.time()
        model = self.model

        # Process detections
        detections = results.xyxy[0]  # Tensor with detections
        if len(detections) > 0:
            # Convert to numpy
            detections = detections.cpu().numpy()
            # Map class IDs to names
            class_names = [model.names[int(cls_id)] for cls_id in detections[:, 5]]
            # Filter detections for Roof and Person
            valid_classes = ['Telhado', 'Pessoa']
            filtered_indices = [i for i, cls_name in enumerate(class_names) if cls_name in valid_classes]
            if filtered_indices:
                # Get filtered detections
                filtered_detections = detections[filtered_indices]
                # Select detection with highest confidence
                best_detection_idx = np.argmax(filtered_detections[:, 4])
                best_detection = filtered_detections[best_detection_idx]
                confidence_score = float(best_detection[4])
                class_id = int(best_detection[5])
                class_name = model.names[class_id]
            else:
                confidence_score = None
                class_name = None
        else:
            confidence_score = None
            class_name = None

        # Handle capturing and saving images
        if self.capturing:
            if current_time - self.last_capture_time >= 2:  # capture_interval
                # Save the image
                filename = f"capture_{int(current_time * 1000)}.jpg"
                filepath = os.path.join(group_capture_dir, filename)
                cv2.imwrite(filepath, self.frame)

                # Update ranking data
                with self.ranking_data_lock:
                    # Load ranking data
                    load_ranking_data()
                    group_entry = get_group_entry(self.group_name)

                    if confidence_score is not None:
                        img_info = {
                            'image_filename': filename,
                            'class': class_name,
                            'confidence': confidence_score
                        }
                        group_entry['images'].append(img_info)
                        # Update top_images and accuracy
                        sorted_images = sorted(group_entry['images'], key=lambda x: x['confidence'], reverse=True)
                        top_images = sorted_images[:3]
                        group_entry['top_images'] = top_images
                        avg_confidence = sum(img['confidence'] for img in top_images) / len(top_images)
                        group_entry['accuracy'] = avg_confidence
                    # Save ranking data
                    save_ranking_data()

                self.last_capture_time = current_time


    def get_frame(self):
        return self.frame




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
    with ranking_data_lock:
        if os.path.exists('ranking.json'):
            try:
                with open('ranking.json', 'r') as f:
                    ranking_data = json.load(f)
                print("ranking_data carregado de ranking.json")
            except json.JSONDecodeError as e:
                print(f"Erro ao decodificar ranking.json: {e}")
                ranking_data = []
                print("Iniciando ranking_data como lista vazia.")
        else:
            ranking_data = []
            print("ranking.json não encontrado. Iniciando ranking_data como lista vazia.")


def save_ranking_data():
    global ranking_data
    with ranking_data_lock:
        try:
            print("Iniciando salvamento do ranking_data...")
            ranking_json_path = os.path.abspath('ranking.json')
            print(f"Caminho absoluto para ranking.json: {ranking_json_path}")
            print(f"Dados a serem salvos em ranking.json: {ranking_data}")
            with open(ranking_json_path, 'w') as f:
                json.dump(ranking_data, f, default=default_serializer)
            print("ranking_data salvo em ranking.json")
        except Exception as e:
            print(f"Erro ao salvar ranking_data: {e}")
            traceback.print_exc()

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
    # Updated to reflect per-group processing status
    group_name = session.get('group_name', 'Anônimo')
    with group_processors_lock:
        processing_active = False
        if group_name in group_processors:
            group_processor = group_processors[group_name]
            processing_active = group_processor.processing_active
    return render_template('live_verification.html', processing_active=processing_active)


@app.route('/start_continuous_capture', methods=['POST'])
def start_continuous_capture():
    """
    Inicia a captura contínua de imagens.
    """
    global continuous_capturing
    continuous_capturing = True

    # Obter o nome do grupo da sessão
    group_name = session.get('group_name', 'Anônimo')

    # Garantir que o diretório do grupo exista
    group_capture_dir = os.path.join('static', 'captures', group_name)
    os.makedirs(group_capture_dir, exist_ok=True)

    def capture_images_continuously():
        global frame
        while continuous_capturing:
            if frame is not None:
                filename = f"capture_{int(time.time() * 1000)}.jpg"
                filepath = os.path.join(group_capture_dir, filename)
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
    group_name = session.get('group_name', 'Anônimo')

    # Ensure group processor exists
    with group_processors_lock:
        if group_name not in group_processors:
            group_processors[group_name] = GroupProcessor(group_name)

    group_processor = group_processors[group_name]
    group_processor.start_processing()

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
    group_name = session.get('group_name', 'Anônimo')

    with group_processors_lock:
        if group_name in group_processors:
            group_processor = group_processors[group_name]
            group_processor.stop_processing()

    return redirect(url_for('live_verification'))

# Video feed
@app.route('/live_feed')
def live_feed():
    group_name = session.get('group_name', 'Anônimo')
    with group_processors_lock:
        if group_name in group_processors:
            group_processor = group_processors[group_name]
            frame = group_processor.get_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    response = make_response(buffer.tobytes())
                    response.headers['Content-Type'] = 'image/jpeg'
                    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                    response.headers['Pragma'] = 'no-cache'
                    response.headers['Expires'] = '0'
                    return response
    # Return 204 No Content if no frame is available
    return '', 204


@app.route('/check_model_status')
def check_model_status():
    group_name = session.get('group_name', 'Anônimo')
    with group_processors_lock:
        if group_name in group_processors:
            group_processor = group_processors[group_name]
            if group_processor.model_loaded and group_processor.processing_active:
                return '', 200
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
        # Obter o nome do grupo da sessão
        group_name = session.get('group_name', 'Anônimo')

        # Garantir que o diretório do grupo exista
        group_capture_dir = os.path.join('static', 'captures', group_name)
        os.makedirs(group_capture_dir, exist_ok=True)

        filename = f"capture_{int(time.time() * 1000)}.jpg"
        filepath = os.path.join(group_capture_dir, filename)
        cv2.imwrite(filepath, frame)
        return "Imagem capturada com sucesso.", 200
    else:
        return "Nenhuma imagem disponível para capturar.", 500

def process_live_video(group_name, capture_images=True, capture_interval=1):
    global processing_active, frame, model, capturing, ranking_data
    last_capture_time = time.time()

    # Garantir que o modelo esteja carregado
    load_group_model(group_name)
    processing_active = True
    capturing = True

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
            img_resp = urllib.request.urlopen(image_url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgnp, -1)

            # Processa a imagem com o modelo
            if model is not None:
                results = model(img)
                frame = np.squeeze(results.render())

                # Filtrar detecções para Telhado e Pessoa
                detections = results.xyxy[0]  # Tensor com as detecções
                if len(detections) > 0:
                    # Converter para numpy
                    detections = detections.cpu().numpy()
                    # Mapear IDs de classe para nomes
                    class_names = [model.names[int(cls_id)] for cls_id in detections[:, 5]]
                    # Filtrar detecções para Telhado e Pessoa
                    valid_classes = ['Telhado', 'Pessoa']
                    filtered_indices = [i for i, cls_name in enumerate(class_names) if cls_name in valid_classes]
                    if filtered_indices:
                        # Obter detecções filtradas
                        filtered_detections = detections[filtered_indices]
                        # Selecionar a detecção com maior confiança
                        best_detection_idx = np.argmax(filtered_detections[:, 4])
                        best_detection = filtered_detections[best_detection_idx]
                        confidence_score = float(best_detection[4])
                        class_id = int(best_detection[5])
                        class_name = model.names[class_id]
                    else:
                        # Se não houver detecções de Telhado ou Pessoa
                        confidence_score = None
                        class_name = None
                else:
                    # Se não houver detecções
                    confidence_score = None
                    class_name = None
            else:
                frame = img
                confidence_score = None
                class_name = None

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

                    # Protege o acesso ao ranking_data
                    with ranking_data_lock:
                        # Carregar os dados atualizados
                        load_ranking_data()
                        # Obter a entrada do grupo no ranking_data
                        group_entry = get_group_entry(group_name)

                        if confidence_score is not None:
                            # Criar o img_info
                            img_info = {
                                'image_filename': filename,
                                'class': class_name,
                                'confidence': confidence_score
                            }
                            # Adicionar à lista de imagens do grupo
                            group_entry['images'].append(img_info)

                            # Ordenar as detecções por confiança
                            sorted_images = sorted(group_entry['images'], key=lambda x: x['confidence'], reverse=True)
                            # Selecionar as 3 melhores
                            top_images = sorted_images[:3]
                            # Calcular a média das 3 melhores confidências
                            avg_confidence = sum(img['confidence'] for img in top_images) / len(top_images)
                            group_entry['accuracy'] = avg_confidence
                            # Atualizar as melhores imagens
                            group_entry['top_images'] = top_images
                        else:
                            # Se não houver detecção válida, não atualiza a acurácia
                            pass

                        # Salvar o ranking_data
                        save_ranking_data()
                    last_capture_time = current_time

        except Exception as e:
            print(f"Erro no processamento do vídeo ao vivo: {e}")
            traceback.print_exc()
        time.sleep(0.1)

def get_group_entry(group_name):
    global ranking_data
    with ranking_data_lock:
        group_entry = next((entry for entry in ranking_data if entry['group'] == group_name), None)
        if group_entry is None:
            group_entry = {
                'group': group_name,
                'accuracy': 0.0,
                'images': [],
                'top_images': []
            }
            ranking_data.append(group_entry)
        return group_entry



@app.route('/start_live_capture', methods=['POST'])
def start_live_capture():
    group_name = session.get('group_name', 'Anônimo')

    with group_processors_lock:
        if group_name not in group_processors:
            group_processors[group_name] = GroupProcessor(group_name)

    group_processor = group_processors[group_name]
    group_processor.capturing = True
    group_processor.start_processing()
    flash('Transmissão ao vivo e captura de imagens iniciadas.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/stop_live_capture', methods=['POST'])
def stop_live_capture():
    group_name = session.get('group_name', 'Anônimo')

    with group_processors_lock:
        if group_name in group_processors:
            group_processor = group_processors[group_name]
            group_processor.capturing = False
            flash('Captura de imagens interrompida.', 'success')
        else:
            flash('Processador de grupo não encontrado.', 'error')
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
        top_images = group_entry.get('top_images', [])
    else:
        images = []
        top_images = []
    return render_template('view_processed_images.html', images=images, top_images=top_images, group_name=group_name)




@app.route('/delete_image', methods=['POST'])
def delete_image():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    group_name = session.get('group_name', 'Anônimo')
    image_filename = request.form.get('image_filename')

    # Caminho completo da imagem
    image_path = os.path.join('static', 'captures', group_name, image_filename)

    # Remover o arquivo de imagem
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Imagem {image_filename} deletada.")

    # Remover informações da imagem do ranking_data
    with ranking_data_lock:
        load_ranking_data()
        group_entry = next((entry for entry in ranking_data if entry['group'] == group_name), None)
        if group_entry and 'images' in group_entry:
            # Remover a imagem da lista
            group_entry['images'] = [img for img in group_entry['images'] if img['image_filename'] != image_filename]
            # Recalcular as top_images
            sorted_images = sorted(
                [img for img in group_entry['images'] if 'confidence' in img and img['confidence'] is not None],
                key=lambda x: x['confidence'],
                reverse=True
            )
            top_images = sorted_images[:3]
            group_entry['top_images'] = top_images
            # Recalcular a acurácia
            if top_images:
                group_entry['accuracy'] = sum(img['confidence'] for img in top_images) / len(top_images)
            else:
                group_entry['accuracy'] = 0.0
            save_ranking_data()

    flash('Imagem deletada com sucesso.', 'success')
    return redirect(url_for('view_processed_images'))



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
    with ranking_data_lock:
        try:
            ranking_json_path = os.path.abspath('ranking.json')
            print(f"Caminho absoluto para ranking.json: {ranking_json_path}")
            print(f"Dados a serem salvos em ranking.json: {ranking_data}")
            with open(ranking_json_path, 'w') as f:
                json.dump(ranking_data, f, default=default_serializer)
            print("ranking_data salvo em ranking.json")
        except Exception as e:
            print(f"Erro ao salvar ranking_data: {e}")
            traceback.print_exc()




def load_ranking_data():
    global ranking_data
    with ranking_data_lock:
        if os.path.exists('ranking.json'):
            try:
                with open('ranking.json', 'r') as f:
                    ranking_data = json.load(f)
                print("ranking_data carregado de ranking.json")
            except json.JSONDecodeError as e:
                print(f"Erro ao decodificar ranking.json: {e}")
                ranking_data = []
                print("Iniciando ranking_data como lista vazia.")
        else:
            ranking_data = []
            print("ranking.json não encontrado. Iniciando ranking_data como lista vazia.")


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
