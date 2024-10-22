# config.py

import os
import json

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24))
    LOG_FILE = os.environ.get('LOG_FILE', 'app.log')
    DEBUG = os.environ.get('DEBUG', 'True').lower() in ['true', '1', 't']
    SETTINGS_FILE = 'settings.json'
    
    @classmethod
    def load_settings(cls):
        if os.path.exists(cls.SETTINGS_FILE):
            try:
                with open(cls.SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                return settings
            except json.JSONDecodeError as e:
                print(f"Erro ao decodificar {cls.SETTINGS_FILE}: {e}")
                return {}
            except Exception as e:
                print(f"Erro ao carregar {cls.SETTINGS_FILE}: {e}")
                return {}
        else:
            # Retorna configurações padrão se o arquivo não existir
            return {
                "camera_url": "http://192.168.1.7/cam-hi.jpg"
            }
    
    @classmethod
    def save_settings(cls, settings):
        try:
            with open(cls.SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
            return True
        except Exception as e:
            print(f"Erro ao salvar {cls.SETTINGS_FILE}: {e}")
            return False
