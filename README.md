# Reconhecimento de Imagens

## Descrição

Este projeto foca na detecção de objetos em imagens, utilizando modelos de reconhecimento, como o YOLOv5. Ele captura imagens por meio de uma câmera acessível via URL, processa essas imagens em tempo real e detecta os objetos presentes.

## Funcionalidades

- Captura de imagens de uma câmera ao vivo via URL.
- YOLOv5: Detecção de objetos em tempo real.
- Interface web moderna com navegação lateral.
- Controle de captura contínua com temporizador.
- Sistema simples de autenticação.
- Visualização e gerenciamento de imagens processadas.
- Registro dos processos e logs de execução.
- Ranking de grupos da competição.

## Front-end

O front-end do projeto é implementado com **HTML5**, **CSS3** e **JavaScript**. Ele é um site responsivo, com os seguintes elementos principais:

- **Menu lateral**: Facilita a navegação entre o dashboard, registro de grupos, ranking de grupos, verificação ao vivo e configurações.
- **Blocos reutilizáveis**: Usando Jinja2 no design do layout, facilita a reutilização de blocos para renderização dinâmica de conteúdo.
- **Integração com Font Awesome**: Utilizado para ícones em botões e seções do projeto, proporcionando uma apresentação moderna.
- **Google Fonts**: Usa a fonte "Roboto" para dar ao texto uma aparência mais leve.

### Estrutura de Páginas

1. **Dashboard**: Acesso rápido para Registro de Grupos, Ranking de Grupos, Verificação ao Vivo e Imagens Processadas.
2. **Registro de Grupos**: Permite o cadastro de novos grupos e o upload de modelos YOLOv5.
3. **Ranking de Grupos**: Exibe o ranking dos grupos com base na precisão das detecções.
4. **Verificação em Tempo Real**: Permite a captura e o processamento ao vivo da câmera, com uma interface para exibir o feed de vídeo e configurar a duração da captura contínua.
5. **Configurações**: Permite alterar a URL da câmera e outras configurações do aplicativo.

### Imagem Fornecida

Uma imagem no formato PNG, com dimensões de 614x511 pixels e modo de cores RGBA, está disponível. Ela pode ser usada como parte da identidade visual do projeto, como logotipo ou ícone (favicon) na interface web. A imagem pode substituir o arquivo `favicon.png` atual, conforme as preferências de design.

### Personalização Visual

- **Logo**: A imagem fornecida pode ser utilizada como logo na tela principal e no menu lateral.
- **Favicon**: A imagem pode ser usada como ícone da aba do navegador, bastando substituir o arquivo em `static/img/favicon.png`.

## Instalação

1. Clone o repositório:
    ```bash
    git clone <url-do-repositorio>
    ```

2. Acesse o diretório do projeto:
    ```bash
    cd Reconhecedor-v
    ```

3. Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate  # Windows
    ```

4. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
    
5. Instale o YoloV5
    ```bash
    git clone https://github.com/ultralytics/yolov5
    ```

## Como Usar

1. Inicie o servidor:
    ```bash
    python app.py
    ```

2. Abra o navegador e acesse `http://localhost:5000`.

## Estrutura do Projeto

- `app.py`: Arquivo principal que executa o aplicativo.
- `app_utils.py`: Funções utilitárias para a aplicação.
- `config.py`: Contém parâmetros de configuração da câmera, modelo e outras inicializações.
- `model_cache.py`: Gerenciamento de cache dos modelos.
- `models/`: Pasta onde os modelos de reconhecimento são armazenados.
- `templates/`: Arquivos HTML para a interface web.
- `static/`: Arquivos estáticos, como CSS, JavaScript e imagens para a interface web.
- `requirements.txt`: Lista de dependências Python com suas versões.

## Licença

Este projeto não possui uma licença permanente definida.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou pull request.

## Autor
**Sandron Oliveira Silva**, Tecnológo em Inteligência Artificial.
