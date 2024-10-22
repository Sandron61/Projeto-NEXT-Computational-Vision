# model_cache.py

import torch
import os
import sys

class ModelCache:
    _cache = {}

    @classmethod
    def get_model(cls, model_path):
        if model_path in cls._cache:
            return cls._cache[model_path]
        else:
            try:
                # Verifica se o caminho do modelo é absoluto ou relativo
                if not os.path.isabs(model_path):
                    # Torna o caminho absoluto baseado no diretório do app.py
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(base_path, model_path)
                
                # Carrega o modelo customizado usando torch.hub.load
                model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')  # 'local' usa o repositório clonado
                model.eval()
                cls._cache[model_path] = model
                return model
            except Exception as e:
                print(f"Erro ao carregar o modelo: {e}")
                return None
