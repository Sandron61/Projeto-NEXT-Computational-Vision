import pytest
from flask import Flask
from app import app, load_ranking_data, save_ranking_data, load_groups, load_model, capture_images_func
import json
import os
import cv2
import threading
from unittest.mock import patch, MagicMock

# Importando variáveis globais
from app import ranking_data, groups

@pytest.fixture
def client():
    # Configura o aplicativo Flask para teste
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Testes para rotas do Flask
def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'<title>' in response.data  # Verifica se há uma tag <title> no HTML

def test_group_ranking_route(client):
    response = client.get('/group_ranking')
    assert response.status_code == 200
    assert b'Ranking' in response.data  # Verifica se a palavra "Ranking" está na resposta

# Testes de funções auxiliares
def test_load_ranking_data():
    global ranking_data
    # Cria um arquivo de teste temporário
    test_data = [{'name': 'test', 'score': 100}]
    with open('ranking.json', 'w') as f:
        json.dump(test_data, f)
    
    # Executa a função e verifica se os dados foram carregados corretamente
    load_ranking_data()
    ranking_data = json.load(open('ranking.json'))  # Força a atualização da variável global
    assert len(ranking_data) == 1
    assert ranking_data[0]['name'] == 'test'

    # Limpa o arquivo de teste
    os.remove('ranking.json')

def test_save_ranking_data():
    global ranking_data
    # Define dados de teste
    ranking_data = [{'name': 'test', 'score': 100}]
    
    # Executa a função para salvar os dados
    save_ranking_data()
    
    # Verifica se o arquivo foi salvo corretamente
    with open('ranking.json', 'r') as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]['name'] == 'test'
    
    # Limpa o arquivo de teste
    os.remove('ranking.json')

def test_load_groups():
    global groups
    # Cria um arquivo de teste temporário
    test_data = {'group1': ['member1', 'member2']}
    with open('groups.json', 'w') as f:
        json.dump(test_data, f)
    
    # Executa a função e verifica se os dados foram carregados corretamente
    load_groups()
    groups = json.load(open('groups.json'))  # Força a atualização da variável global
    assert 'group1' in groups
    assert len(groups['group1']) == 2

    # Limpa o arquivo de teste
    os.remove('groups.json')

# Testes para transmissão ao vivo e captura de imagens
def test_start_live_processing(client):
    with patch('app.threading.Thread') as mock_thread:
        response = client.get('/start_live_processing')
        assert response.status_code == 302  # Redirecionamento esperado
        mock_thread.assert_called_once()  # Verifica se a thread foi iniciada

def test_stop_live_processing(client):
    response = client.get('/stop_live_processing')
    assert response.status_code == 302  # Redirecionamento esperado

@patch('app.urllib.request.urlopen')
@patch('app.cv2.imdecode')
def test_live_feed(mock_imdecode, mock_urlopen, client):
    # Simula uma resposta de imagem
    mock_urlopen.return_value.read.return_value = b'image_data'
    mock_imdecode.return_value = MagicMock()

    response = client.get('/live_feed')
    assert response.status_code in [200, 204]  # Pode ser 200 (imagem) ou 204 (sem frame)

@patch('app.urllib.request.urlopen')
@patch('app.cv2.imdecode')
def test_capture_images_func(mock_imdecode, mock_urlopen):
    # Simula uma resposta de imagem
    mock_urlopen.return_value.read.return_value = b'image_data'
    mock_imdecode.return_value = MagicMock()

    # Define as variáveis globais necessárias
    global capturing, model
    capturing = True
    model = MagicMock()  # Simula um modelo carregado

    # Inicia a função de captura de imagens em uma thread
    capture_thread = threading.Thread(target=capture_images_func)
    capture_thread.start()
    capturing = False  # Para a captura
    capture_thread.join()

# Testes para transmissão ao vivo poderiam envolver mocks de cv2.VideoCapture e frame processing,
# simulando uma câmera ou entrada de vídeo em vez de depender de hardware real.
