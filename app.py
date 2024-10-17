import os
from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
from werkzeug.utils import secure_filename
import cv2
import torch
import numpy as np
import urllib.request
import base64

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Caminho para salvar os modelos enviados
MODEL_UPLOAD_FOLDER = 'models'
app.config['MODEL_UPLOAD_FOLDER'] = MODEL_UPLOAD_FOLDER

# Variável global para o caminho do modelo atual
current_model_path = os.path.join(MODEL_UPLOAD_FOLDER, 'best.pt')

# Verifica se a pasta de modelos existe, caso contrário, cria
if not os.path.exists(MODEL_UPLOAD_FOLDER):
    os.makedirs(MODEL_UPLOAD_FOLDER)

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

# Função para processar imagens ao vivo
def process_live_images():
    # Verifica se o modelo existe
    if not os.path.exists(current_model_path):
        return None

    # URL da imagem ao vivo (substitua pelo seu endereço)
    image_url = 'http://172.20.10.3/cam-hi.jpg'  # TROQUE PELO LINK GERADO NO MONITOR SERIAL

    # Carrega o modelo
    model = torch.hub.load('ultralytics/yolov5', 'custom', current_model_path, force_reload=True)
    model.conf = 0.6

    try:
        # Captura a imagem da URL
        img_resp = urllib.request.urlopen(url=image_url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        # Processa a imagem
        results = model(im)
        frame = np.squeeze(results.render())

        # Codifica a imagem para base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = buffer.tobytes()
        frame_base64 = base64.b64encode(frame_encoded).decode('utf-8')

        return frame_base64
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return None

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

# Rota para obter a imagem ao vivo via AJAX
@app.route('/get_live_image')
def get_live_image():
    if 'username' in session:
        result = process_live_images()
        if result:
            return jsonify({'result': result})
        else:
            return jsonify({'error': 'Não foi possível processar a imagem ao vivo.'})
    else:
        return jsonify({'error': 'Usuário não autenticado.'})

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

# Página de ranking de grupos
@app.route('/group_ranking')
def group_ranking():
    if 'username' in session:
        # Exemplo de dados de ranking
        ranking_data = [
            {'group': 'Grupo A', 'accuracy': 95},
            {'group': 'Grupo B', 'accuracy': 90},
            {'group': 'Grupo C', 'accuracy': 85},
        ]
        return render_template('group_ranking.html', ranking_data=ranking_data)
    else:
        return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
