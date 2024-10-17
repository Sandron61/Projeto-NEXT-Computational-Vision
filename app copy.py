import os
from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
from werkzeug.utils import secure_filename
import cv2
import torch
import numpy as np
import urllib.request
import base64
import threading
import time
import json
import pathlib
from pathlib import Path
import logging


# Configurações de logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

# Correção para compatibilidade de path
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Caminho para salvar os modelos enviados
MODEL_UPLOAD_FOLDER = 'models'
app.config['MODEL_UPLOAD_FOLDER'] = MODEL_UPLOAD_FOLDER

# Pasta para armazenar capturas temporárias
CAPTURE_FOLDER = 'captures'
Path(CAPTURE_FOLDER).mkdir(exist_ok=True)

# Arquivo para armazenar o ranking de grupos
RANKING_FILE = 'ranking.json'

# Inicializa o ranking se o arquivo não existir
if not os.path.exists(RANKING_FILE):
    with open(RANKING_FILE, 'w') as f:
        json.dump({}, f)

# Verifica se a pasta de modelos existe, caso contrário, cria
if not os.path.exists(MODEL_UPLOAD_FOLDER):
    os.makedirs(MODEL_UPLOAD_FOLDER)

# Variável global para o caminho do modelo atual
current_model_path = os.path.join(MODEL_UPLOAD_FOLDER, 'best.pt')


model = None

# Evento para sincronizar o início da captura
start_capture_event = threading.Event()

def load_model():
    global model
    if not os.path.exists(current_model_path):
        logging.error("Modelo não encontrado no caminho especificado.")
        return False
    try:
        logging.debug("Carregando o modelo YOLOv5")
        model = torch.hub.load('ultralytics/yolov5', 'custom', current_model_path, force_reload=True)
        model.conf = 0.6
        logging.debug("Modelo carregado com sucesso")
        return True
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo: {e}")
        return False

if not load_model():
    logging.critical("Falha ao carregar o modelo. A aplicação será encerrada.")
    exit(1)

# Variável global para armazenar a última imagem processada
latest_image = None
image_lock = threading.Lock()

def process_live_images():
    global latest_image, start_capture_event  # Certifique-se de declarar 'start_capture_event' como global
    try:
        logging.debug("Aguardando permissão para iniciar o processamento de imagens...")
        # Espera até que o evento seja liberado
        start_capture_event.wait()

        logging.debug("Iniciando processamento contínuo de imagens.")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        while True:
            try:
                logging.debug("Capturando a imagem da URL")
                img_resp = urllib.request.urlopen(url='http://192.168.1.7/cam-hi.jpg')
                imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                im = cv2.imdecode(imgnp, -1)
                logging.debug("Imagem capturada e decodificada com sucesso")

                # Redimensionar a imagem para otimizar o processamento (opcional)
                # im = cv2.resize(im, (640, 480))

                # Processa a imagem com o modelo YOLOv5
                logging.debug("Processando a imagem com o modelo")
                results = model(im, size=640)  # Ajuste o tamanho conforme necessário
                confidences = results.pandas().xyxy[0]['confidence'].tolist()
                avg_conf = np.mean(confidences) if confidences else 0
                logging.debug(f"Confiança média das detecções: {avg_conf:.2f}")

                # Salva a imagem com um nome único
                timestamp = int(time.time())
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join(CAPTURE_FOLDER, filename)
                cv2.imwrite(filepath, im)
                logging.debug(f"Captura {filename} salva com confiança {avg_conf:.2f}")

                # Codifica a imagem para Base64
                logging.debug("Codificando a imagem para base64")
                _, buffer = cv2.imencode('.jpg', im)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                logging.debug("Imagem codificada com sucesso")

                # Atualiza a variável global com thread safety
                with image_lock:
                    latest_image = img_base64

            except Exception as e:
                logging.error(f"Erro durante captura e processamento contínuo: {e}")

            # Intervalo entre capturas (ajuste conforme necessário)
            time.sleep(2)
    except Exception as e:
        logging.error(f"Erro na thread de processamento contínuo: {e}")

# Iniciar a thread de processamento contínuo
thread = threading.Thread(target=process_live_images, daemon=True)
thread.start()

# Função para sinalizar o início da captura após o setup completo
def setup_complete():
    logging.debug("Configurações completas. Iniciando captura de imagens.")
    start_capture_event.set()

# Rota para obter a imagem ao vivo via AJAX
@app.route('/get_live_image')
def get_live_image():
    if 'username' in session:
        with image_lock:
            if latest_image:
                return jsonify({'result': latest_image})
            else:
                return jsonify({'error': 'Imagem ainda não processada.'})
    else:
        return jsonify({'error': 'Usuário não autenticado.'})



            
# Página de login
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'teste' and password == 'teste123':
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = 'Credenciais inválidas'
            return render_template('login.html', error=error)
    return render_template('login.html')

# Página de logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Página do dashboard
@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html')
    else:
        return redirect(url_for('login'))

# Página de upload do modelo
@app.route('/upload_model', methods=['GET', 'POST'])
def upload_model():
    if 'username' in session:
        if request.method == 'POST':
            if 'model_file' not in request.files:
                flash('Nenhum arquivo selecionado')
                return redirect(request.url)
            file = request.files['model_file']
            if file.filename == '':
                flash('Nenhum arquivo selecionado')
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['MODEL_UPLOAD_FOLDER'], filename))
                global current_model_path
                current_model_path = os.path.join(app.config['MODEL_UPLOAD_FOLDER'], filename)
                flash('Modelo enviado com sucesso')
                return redirect(url_for('upload_model'))
        # Passa o nome do modelo atual para o template
        current_model = os.path.basename(current_model_path) if os.path.exists(current_model_path) else None
        return render_template('upload_model.html', current_model=current_model)
    else:
        return redirect(url_for('login'))



@app.route('/view_results')
def view_results():
    if 'username' in session:
        try:
            with open(RANKING_FILE, 'r') as f:
                ranking = json.load(f)
        except Exception as e:
            flash(f"Erro ao ler o arquivo de ranking: {e}", "error")
            ranking = {}

        # Ordena o ranking
        ranking_sorted = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
        ranking_data = [{'group': grp, 'points': pts} for grp, pts in ranking_sorted]

        return render_template('view_results.html', ranking_data=ranking_data)
    else:
        return redirect(url_for('login'))

# Página de verificação ao vivo
@app.route('/live_verification')
def live_verification():
    if 'username' in session:
        result = process_live_images()
        if result:
            return render_template('live_verification.html', result=result)
        else:
            error = 'Não foi possível processar a imagem ao vivo.'
            return render_template('live_verification.html', error=error)
    else:
        return redirect(url_for('login'))











    
# Outras rotas (exemplo: cadastro de grupos)
@app.route('/register_group', methods=['GET', 'POST'])
def register_group():
    if 'username' in session:
        if request.method == 'POST':
            # Lógica para cadastrar grupo
            group_name = request.form['group_name']
            # Salve o grupo no banco de dados (não implementado aqui)
            flash(f'Grupo "{group_name}" cadastrado com sucesso!')
            return redirect(url_for('dashboard'))
        return render_template('register_group.html')
    else:
        return redirect(url_for('login'))

@app.route('/group_ranking')
def group_ranking():
    if 'username' in session:
        try:
            with open(RANKING_FILE, 'r') as f:
                ranking = json.load(f)
        except Exception as e:
            flash(f"Erro ao ler o arquivo de ranking: {e}", "error")
            ranking = {}

        # Ordena o ranking
        ranking_sorted = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
        ranking_data = [{'group': grp, 'accuracy': pts} for grp, pts in ranking_sorted]

        return render_template('group_ranking.html', ranking_data=ranking_data)
    else:
        return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
