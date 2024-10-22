# app.py

from flask import Flask, render_template, request, redirect, url_for, session, make_response, flash
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
import logging
import sys  # Importação adicionada

from config import Config
from app_utils import allowed_file, secure_filename_custom, TryExcept  # Importação atualizada
from model_cache import ModelCache  # Importação atualizada

# Configurações de caminho para sistemas Windows
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
app.config.from_object(Config)

# Configuração de Logging
logging.basicConfig(
    filename=app.config['LOG_FILE'],
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Variáveis globais
groups = {}
ranking_data = []
ranking_data_lock = threading.RLock()
group_processors = {}
group_processors_lock = threading.Lock()

# Carregar configurações iniciais
settings = Config.load_settings()
app.config['CAMERA_URL'] = settings.get('camera_url', 'http://192.168.1.7/cam-hi.jpg')

# Adiciona o diretório yolov5 ao PYTHONPATH antes do diretório atual
yolov5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)  # Insere no início para priorizar yolov5

# Funções de Serialização
def default_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)

# Classe para Processamento de Grupos
class GroupProcessor:
    def __init__(self, group_name):
        self.group_name = group_name
        self.model = None
        self.stop_event = threading.Event()  # Evento para sinalizar a parada
        self.model_loaded = False
        self.processing_active = False
        self.capturing = False
        self.frame = None
        self.capture_thread = None
        self.last_capture_time = time.time()
        self.camera_url = app.config['CAMERA_URL']
        self.group_capture_dir = os.path.join('static', 'captures', self.group_name)
        os.makedirs(self.group_capture_dir, exist_ok=True)
        self.load_model()

    def load_model(self):
        global groups
        model_path = groups.get(self.group_name, {}).get('model')
        if model_path and os.path.exists(model_path):
            logging.info(f"Tentando carregar o modelo para o grupo '{self.group_name}' a partir de '{model_path}'")
            try:
                self.model = ModelCache.get_model(model_path)
                if self.model:
                    self.model_loaded = True
                    logging.info(f"Modelo carregado com sucesso para o grupo '{self.group_name}'")
                else:
                    self.model_loaded = False
                    logging.error(f"Falha ao carregar o modelo para o grupo '{self.group_name}' a partir de '{model_path}'")
            except Exception as e:
                self.model_loaded = False
                logging.error(f"Erro ao carregar o modelo para o grupo '{self.group_name}': {e}")
        else:
            logging.error(f"Modelo para o grupo '{self.group_name}' não encontrado em '{model_path}'.")
            self.model_loaded = False

    def start_processing(self):
        if not self.processing_active:
            self.processing_active = True
            self.capturing = True
            self.stop_event.clear()  # Limpa o evento de parada
            self.capture_thread = threading.Thread(target=self.process_live_video, daemon=True)
            self.capture_thread.start()
            logging.info(f"Iniciado processamento para o grupo {self.group_name}.")
        else:
            logging.warning(f"Processamento já iniciado para o grupo {self.group_name}.")



    def stop_processing(self):
        if self.processing_active:
            self.processing_active = False
            self.capturing = False
            self.stop_event.set()  # Sinaliza para a thread parar
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=5)
                if self.capture_thread.is_alive():
                    logging.warning(f"Thread de captura não parou dentro do tempo para o grupo {self.group_name}.")
                else:
                    logging.info(f"Thread de captura parada para o grupo {self.group_name}.")
            logging.info(f"Processamento parado para o grupo {self.group_name}.")
        else:
            logging.warning(f"Processamento não está ativo para o grupo {self.group_name}.")


    def process_live_video(self):
        if not self.model_loaded:
            logging.error(f"Modelo não carregado para o grupo {self.group_name}.")
            return

        while not self.stop_event.is_set():
            try:
                logging.debug(f"Processando vídeo ao vivo para o grupo {self.group_name}.")
                # Captura a imagem da câmera
                self.camera_url = app.config['CAMERA_URL']
                img_resp = urllib.request.urlopen(self.camera_url, timeout=5)
                imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                img = cv2.imdecode(imgnp, -1)

                results = self.model(img)
                self.frame = np.squeeze(results.render())

                self.process_detections(results, img)
                logging.debug(f"Detecções processadas para o grupo {self.group_name}.")
            except Exception as e:
                logging.error(f"Erro no processamento de vídeo ao vivo para o grupo {self.group_name}: {e}")
                traceback.print_exc()
            time.sleep(0.1)



    def process_detections(self, results, img):
        current_time = time.time()
        detections = results.xyxy[0]  # Tensor com detecções

        if len(detections) > 0:
            detections = detections.cpu().numpy()
            class_names = [self.model.names[int(cls_id)] for cls_id in detections[:, 5]]
            valid_classes = ['Telhado', 'Pessoa']
            filtered_indices = [i for i, cls_name in enumerate(class_names) if cls_name in valid_classes]

            if filtered_indices:
                best_detection = detections[filtered_indices[np.argmax(detections[filtered_indices, 4])]]
                confidence_score = float(best_detection[4])
                class_name = self.model.names[int(best_detection[5])]

                if self.capturing and (current_time - self.last_capture_time >= 2):  # Intervalo de captura
                    filename = f"capture_{int(current_time * 1000)}.jpg"
                    filepath = os.path.join(self.group_capture_dir, filename)
                    cv2.imwrite(filepath, self.frame)

                    # Atualizar ranking data
                    with ranking_data_lock:
                        load_ranking_data()
                        group_entry = get_group_entry(self.group_name)

                        img_info = {'image_filename': filename, 'class': class_name, 'confidence': confidence_score}
                        group_entry['images'].append(img_info)

                        sorted_images = sorted(group_entry['images'], key=lambda x: x['confidence'], reverse=True)
                        top_images = sorted_images[:3]
                        group_entry['top_images'] = top_images
                        group_entry['accuracy'] = sum(img['confidence'] for img in top_images) / len(top_images)

                        save_ranking_data()

                    self.last_capture_time = current_time
                    logging.info(f"Imagem capturada e salva: {filename} para o grupo {self.group_name}.")

    def get_frame(self):
        return self.frame

    def start_continuous_capture(self, duration):
        """Inicia a captura contínua em uma thread separada."""
        if not self.capturing:
            self.capturing = True
            self.stop_event.clear()  # Limpa o evento de parada
            self.capture_thread = threading.Thread(target=self._capture_images_for_duration, args=(duration,), daemon=True)
            self.capture_thread.start()
            logging.info(f"Captura contínua iniciada para o grupo {self.group_name} por {duration} segundos.")
        else:
            logging.warning("Captura contínua já está em andamento.")

    def _capture_images_for_duration(self, duration):
        """Método interno para capturar imagens por uma duração especificada."""
        start_time = time.time()
        while not self.stop_event.is_set() and (time.time() - start_time) < duration:
            # Captura a imagem
            self.capture_image()
            
            # Dorme em intervalos pequenos para permitir uma parada mais rápida
            for _ in range(40):  # 40 * 0.05 = 2 segundos
                if self.stop_event.is_set():
                    break
                time.sleep(0.05)  # 0.05 segundos
        self.capturing = False
        logging.info(f"Captura contínua finalizada para o grupo {self.group_name}.")


    def capture_image(self):
        try:
            # Atualizar a URL da câmera antes de capturar
            self.camera_url = app.config['CAMERA_URL']
            img_resp = urllib.request.urlopen(self.camera_url, timeout=3)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgnp, -1)

            if not self.model_loaded:
                logging.error(f"Modelo não carregado para o grupo {self.group_name}.")
                return

            results = self.model(img)
            self.frame = np.squeeze(results.render())

            filename = f"capture_{int(time.time() * 1000)}.jpg"
            filepath = os.path.join(self.group_capture_dir, filename)
            cv2.imwrite(filepath, self.frame)

            logging.info(f"Imagem capturada e salva: {filename} para o grupo {self.group_name}.")

            # Atualizar ranking data
            with ranking_data_lock:
                load_ranking_data()
                group_entry = get_group_entry(self.group_name)

                img_info = {'image_filename': filename, 'class': None, 'confidence': None}  # Adapte conforme necessário
                group_entry['images'].append(img_info)

                sorted_images = sorted(group_entry['images'], key=lambda x: x.get('confidence', 0), reverse=True)
                top_images = sorted_images[:3]
                group_entry['top_images'] = top_images
                group_entry['accuracy'] = sum(img.get('confidence', 0) for img in top_images) / len(top_images) if top_images else 0.0

                save_ranking_data()

        except Exception as e:
            logging.error(f"Erro ao capturar imagem para o grupo {self.group_name}: {e}")
            traceback.print_exc()

    def stop_continuous_capture(self):
        """Para a captura contínua de maneira graciosa."""
        if self.capturing:
            self.capturing = False
            self.stop_event.set()  # Sinaliza para a thread parar
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=5)
                if self.capture_thread.is_alive():
                    logging.warning(f"Thread de captura contínua não parou dentro do tempo para o grupo {self.group_name}.")
                else:
                    logging.info(f"Thread de captura contínua parada para o grupo {self.group_name}.")
            logging.info(f"Captura contínua parada para o grupo {self.group_name}.")
        else:
            logging.warning("Nenhuma captura contínua está em andamento.")


# Funções para Gerenciamento de Dados
def load_ranking_data():
    global ranking_data
    with ranking_data_lock:
        try:
            if os.path.exists('ranking.json'):
                with open('ranking.json', 'r') as f:
                    ranking_data = json.load(f)
                logging.info("ranking_data carregado de ranking.json")
            else:
                ranking_data = []
                logging.warning("ranking.json não encontrado. Iniciando ranking_data como lista vazia.")
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar ranking.json: {e}")
            ranking_data = []
        except Exception as e:
            logging.critical(f"Erro inesperado ao carregar ranking_data: {e}")
            ranking_data = []

def save_ranking_data():
    global ranking_data
    with ranking_data_lock:
        try:
            with open('ranking.json', 'w') as f:
                json.dump(ranking_data, f, default=default_serializer, indent=4)
            logging.info("ranking_data salvo em ranking.json")
        except Exception as e:
            logging.error(f"Erro ao salvar ranking_data: {e}")
            traceback.print_exc()

def load_groups():
    global groups
    if os.path.exists('groups.json'):
        try:
            with open('groups.json', 'r') as f:
                groups = json.load(f)
            logging.info(f"Grupos carregados de groups.json: {groups}")
        except Exception as e:
            logging.error(f"Erro ao carregar grupos de groups.json: {e}")
            traceback.print_exc()
            groups = {}
    else:
        groups = {}
        logging.warning("Arquivo groups.json não encontrado. Iniciando com dicionário de grupos vazio.")

def save_groups():
    global groups
    try:
        with open('groups.json', 'w') as f:
            json.dump(groups, f, indent=4)
        logging.info("Grupos salvos com sucesso em groups.json")
    except Exception as e:
        logging.error(f"Erro ao salvar grupos em groups.json: {e}")
        traceback.print_exc()

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

# Rotas de Autenticação
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        # Autenticação simples
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username == 'admin' and password == 'password':  # Substitua por um método de autenticação mais seguro
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('select_group'))  # Redireciona para selecionar grupo após login
        else:
            error = 'Usuário ou senha inválidos'
            logging.warning(f"Tentativa de login falhada para usuário: {username}")
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    flash('Você foi desconectado com sucesso.', 'success')
    return redirect(url_for('login'))

# Rota do Dashboard
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

# Rota para Registrar Grupo
@app.route('/register_group', methods=['GET', 'POST'])
def register_group():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        group_name = request.form.get('group_name', '').strip()
        if not group_name:
            flash('O nome do grupo não pode estar vazio.', 'error')
            return redirect(url_for('register_group'))
        
        if 'model_file' not in request.files:
            flash('Nenhum arquivo de modelo enviado.', 'error')
            return redirect(url_for('register_group'))
        
        model_file = request.files['model_file']
        if model_file.filename == '':
            flash('Nenhum arquivo selecionado.', 'error')
            return redirect(url_for('register_group'))
        
        if model_file and allowed_file(model_file.filename):
            filename = secure_filename_custom(model_file.filename)
            group_dir = os.path.join('models', group_name)
            os.makedirs(group_dir, exist_ok=True)
            model_path = os.path.join(group_dir, 'model.pt')
            model_file.save(model_path)
            
            if os.path.exists(model_path):
                logging.info(f"Modelo salvo com sucesso em: {model_path}")
                load_groups()
                groups[group_name] = {'model': model_path}
                save_groups()
                session['group_name'] = group_name
                session['model_name'] = os.path.basename(model_path)
                flash(f'Grupo {group_name} registrado com sucesso e modelo carregado.', 'success')
                return redirect(url_for('dashboard'))
            else:
                logging.error(f"Erro ao salvar o modelo para o grupo {group_name}")
                flash('Erro ao salvar o arquivo do modelo.', 'error')
                return redirect(url_for('register_group'))
        else:
            flash('Tipo de arquivo inválido. Por favor, envie um arquivo .pt.', 'error')
            return redirect(url_for('register_group'))
    
    return render_template('register_group.html')

# Rota para Selecionar Grupo
@app.route('/select_group', methods=['GET', 'POST'])
def select_group():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    load_groups()  # Carrega os grupos disponíveis

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

# Rota para Verificação ao Vivo
@app.route('/live_verification')
def live_verification():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    group_name = session.get('group_name', 'Anônimo')
    with group_processors_lock:
        processing_active = False
        if group_name in group_processors:
            group_processor = group_processors[group_name]
            processing_active = group_processor.processing_active
    return render_template('live_verification.html', processing_active=processing_active)

# Rota para Iniciar Processamento ao Vivo
@app.route('/start_live_processing')
def start_live_processing():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    group_name = session.get('group_name', 'Anônimo')
    if group_name == 'Anônimo':
        flash('Selecione um grupo antes de iniciar o processamento ao vivo.', 'error')
        return redirect(url_for('select_group'))

    with group_processors_lock:
        if group_name not in group_processors:
            group_processors[group_name] = GroupProcessor(group_name)
    
    group_processor = group_processors[group_name]
    group_processor.start_processing()
    flash('Processamento ao vivo iniciado.', 'success')
    return redirect(url_for('live_verification'))

# Rota para Parar Processamento ao Vivo
@app.route('/stop_live_processing')
def stop_live_processing():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    group_name = session.get('group_name', 'Anônimo')
    if group_name == 'Anônimo':
        flash('Nenhum grupo selecionado para parar o processamento.', 'error')
        return redirect(url_for('select_group'))

    with group_processors_lock:
        if group_name in group_processors:
            group_processor = group_processors[group_name]
            group_processor.stop_processing()
            flash('Processamento ao vivo parado.', 'success')
        else:
            flash('Processador de grupo não encontrado.', 'error')
    return redirect(url_for('live_verification'))

# Feed de Vídeo ao Vivo
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
    return '', 204

# Rota para Checar Status do Modelo/Processamento
@app.route('/check_model_status')
def check_model_status():
    group_name = session.get('group_name', 'Anônimo')
    with group_processors_lock:
        if group_name in group_processors:
            group_processor = group_processors[group_name]
            if group_processor.model_loaded and group_processor.processing_active:
                return '', 200
    return '', 503

# Rota para Capturar Imagem ao Vivo
@app.route('/capture_live_image', methods=['POST'])
def capture_live_image():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    group_name = session.get('group_name', 'Anônimo')
    if not group_name or group_name == 'Anônimo':
        return "Nenhum grupo selecionado.", 400

    with group_processors_lock:
        if group_name in group_processors:
            group_processor = group_processors[group_name]
            if group_processor.frame is not None:
                filename = f"capture_{int(time.time() * 1000)}.jpg"
                filepath = os.path.join(group_processor.group_capture_dir, filename)
                cv2.imwrite(filepath, group_processor.frame)
                logging.info(f"Imagem capturada e salva: {filename} para o grupo {group_name}.")

                # Atualizar ranking data
                with ranking_data_lock:
                    load_ranking_data()
                    group_entry = get_group_entry(group_name)

                    img_info = {'image_filename': filename, 'class': None, 'confidence': None}
                    group_entry['images'].append(img_info)

                    sorted_images = sorted(group_entry['images'], key=lambda x: x.get('confidence', 0), reverse=True)
                    top_images = sorted_images[:3]
                    group_entry['top_images'] = top_images
                    group_entry['accuracy'] = sum(img.get('confidence', 0) for img in top_images) / len(top_images) if top_images else 0.0

                    save_ranking_data()

                return "Imagem capturada com sucesso.", 200
            else:
                return "Nenhuma imagem disponível para capturar.", 500
        else:
            return 'Processador de grupo não encontrado', 400

# Rota para Iniciar Captura Contínua
@app.route('/start_continuous_capture', methods=['POST'])
def start_continuous_capture_route():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    data = request.get_json()
    if not data:
        return 'Bad Request: No JSON data received', 400

    duration = data.get('duration', 60)
    group_name = session.get('group_name', 'Anônimo')

    if not group_name or group_name == 'Anônimo':
        return 'Nenhum grupo selecionado', 400

    with group_processors_lock:
        if group_name in group_processors:
            group_processor = group_processors[group_name]
            group_processor.start_continuous_capture(duration)
            flash('Captura contínua iniciada.', 'success')
            return '', 200
        else:
            return 'Processador de grupo não encontrado', 400

# Rota para Parar Captura Contínua
@app.route('/stop_continuous_capture', methods=['POST'])
def stop_continuous_capture_route():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    group_name = session.get('group_name', 'Anônimo')
    
    if not group_name or group_name == 'Anônimo':
        return 'Nenhum grupo selecionado', 400

    with group_processors_lock:
        if group_name in group_processors:
            group_processor = group_processors[group_name]
            group_processor.stop_continuous_capture()
            flash('Captura contínua parada.', 'success')
            return '', 200
        else:
            return 'Processador de grupo não encontrado', 400

# Rota para Visualizar Imagens Processadas
@app.route('/view_processed_images')
def view_processed_images():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    group_name = session.get('group_name', 'Anônimo')
    if not group_name or group_name == 'Anônimo':
        flash('Selecione um grupo para visualizar as imagens processadas.', 'error')
        return redirect(url_for('select_group'))

    load_ranking_data()
    group_entry = next((entry for entry in ranking_data if entry['group'] == group_name), None)
    if group_entry and 'images' in group_entry:
        images = group_entry['images']
        top_images = group_entry.get('top_images', [])
    else:
        images = []
        top_images = []
    return render_template('view_processed_images.html', images=images, top_images=top_images, group_name=group_name)

# Rota para Deletar Imagem
@app.route('/delete_image', methods=['POST'])
def delete_image():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    group_name = session.get('group_name', 'Anônimo')
    image_filename = request.form.get('image_filename')

    if not group_name or group_name == 'Anônimo':
        flash('Selecione um grupo antes de deletar imagens.', 'error')
        return redirect(url_for('select_group'))

    image_path = os.path.join('static', 'captures', group_name, image_filename)

    if os.path.exists(image_path):
        try:
            os.remove(image_path)
            logging.info(f"Imagem {image_filename} deletada para o grupo {group_name}.")
        except Exception as e:
            logging.error(f"Erro ao deletar a imagem {image_filename}: {e}")
            flash('Erro ao deletar a imagem.', 'error')
            return redirect(url_for('view_processed_images'))
    else:
        logging.warning(f"Tentativa de deletar imagem inexistente: {image_filename} para o grupo {group_name}.")

    with ranking_data_lock:
        load_ranking_data()
        group_entry = next((entry for entry in ranking_data if entry['group'] == group_name), None)
        if group_entry and 'images' in group_entry:
            group_entry['images'] = [img for img in group_entry['images'] if img['image_filename'] != image_filename]
            sorted_images = sorted(
                [img for img in group_entry['images'] if img.get('confidence') is not None],
                key=lambda x: x['confidence'],
                reverse=True
            )
            top_images = sorted_images[:3]
            group_entry['top_images'] = top_images
            group_entry['accuracy'] = sum(img['confidence'] for img in top_images) / len(top_images) if top_images else 0.0
            save_ranking_data()

    flash('Imagem deletada com sucesso.', 'success')
    return redirect(url_for('view_processed_images'))

# Rota para Exibir Ranking de Grupos
@app.route('/group_ranking')
def group_ranking():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    load_ranking_data()
    sorted_ranking = sorted(ranking_data, key=lambda x: x['accuracy'], reverse=True)
    return render_template('group_ranking.html', ranking_data=sorted_ranking)

# Rota Alternativa para Ranking de Grupos
@app.route('/view_results')
def view_results():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    load_ranking_data()
    sorted_ranking = sorted(ranking_data, key=lambda x: x['accuracy'], reverse=True)
    return render_template('view_results.html', ranking_data=sorted_ranking)

# Rota para Configurações
@app.route('/settings', methods=['GET', 'POST'])
def settings_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    current_settings = Config.load_settings()
    
    if request.method == 'POST':
        new_camera_url = request.form.get('camera_url', '').strip()
        if not new_camera_url:
            flash('A URL da câmera não pode estar vazia.', 'error')
            return redirect(url_for('settings_page'))
        
        # Atualizar as configurações
        current_settings['camera_url'] = new_camera_url
        if Config.save_settings(current_settings):
            app.config['CAMERA_URL'] = new_camera_url
            # Atualizar a URL da câmera nos processadores de grupo ativos
            with group_processors_lock:
                for processor in group_processors.values():
                    processor.camera_url = new_camera_url
            flash('Configurações atualizadas com sucesso.', 'success')
        else:
            flash('Falha ao salvar as configurações.', 'error')
        
        return redirect(url_for('settings_page'))
    
    return render_template('settings.html', camera_url=current_settings.get('camera_url', ''))

# Inicialização do Aplicativo
if __name__ == '__main__':
    # Verificar e criar diretórios necessários
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/captures', exist_ok=True)
    
    # Carregar dados iniciais
    load_groups()
    load_ranking_data()
    
    # Iniciar o servidor Flask
    app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])
