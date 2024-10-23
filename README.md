
# Reconhecedor de Imagens

## Descrição

Este projeto é uma aplicação para reconhecimento de objetos em imagens, utilizando modelos de detecção como o YOLOv5. A aplicação captura imagens de uma câmera acessível via URL, processa essas imagens em tempo real e identifica os objetos presentes.

## Funcionalidades

- Captura de imagens de uma câmera ao vivo via URL.
- Reconhecimento de objetos em tempo real utilizando o modelo YOLOv5.
- Interface web moderna com navegação lateral.
- Controle de captura contínua com temporizador.
- Sistema de autenticação simples.
- Visualização e gerenciamento de imagens processadas.
- Registro de logs de processamento.

## Front-end

O front-end do projeto é construído utilizando **HTML5**, **CSS3**, e **JavaScript**. Ele é responsivo e contém os seguintes elementos principais:

- **Menu lateral**: Navegação fácil entre as funcionalidades, como Dashboard, Cadastro de Grupos, Ranking de Grupos, Verificação ao Vivo e Configurações.
- **Blocos reutilizáveis**: O layout foi projetado utilizando o Jinja2, permitindo a reutilização de blocos para renderização dinâmica de conteúdo.
- **Integração com Font Awesome**: Para ícones em botões e seções do projeto, garantindo uma aparência moderna.
- **Google Fonts**: Usa a fonte "Roboto" para dar uma aparência mais leve ao texto.

### Estrutura de Páginas

1. **Dashboard**: Exibe atalhos para o Cadastro de Grupos, Ranking de Grupos, Verificação ao Vivo, e Imagens Processadas.
2. **Cadastro de Grupos**: Permite o registro de novos grupos e o upload de modelos YOLOv5.
3. **Ranking de Grupos**: Exibe a classificação dos grupos com base na acurácia das detecções.
4. **Verificação ao Vivo**: Permite iniciar e parar a captura e processamento ao vivo da câmera, com uma interface para exibir o feed de vídeo e configurar a duração da captura contínua.
5. **Configurações**: Permite alterar a URL da câmera e outras configurações da aplicação.

### Imagem Fornecida

Uma imagem no formato **PNG**, com dimensões **614x511 pixels** e modo de cores **RGBA**, foi fornecida. Esta imagem pode ser usada como parte da identidade visual do projeto, como um **logotipo** ou **ícone de favicon** na interface web. Atualmente, o arquivo `favicon.png` referenciado no projeto pode ser substituído por esta imagem, dependendo das preferências de design.

### Customização Visual

- **Logo**: A imagem enviada pode ser usada como logo na página inicial e no menu lateral.
- **Favicon**: A imagem também pode ser utilizada como ícone da aba do navegador, bastando substituí-la no caminho `static/img/favicon.png`.

## Instalação

1. Clone este repositório:
    ```bash
    git clone <url-do-repositorio>
    ```

2. Navegue até o diretório do projeto:
    ```bash
    cd Reconhecedor-v0.1
    ```

3. Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate  # Windows
    ```

4. Instale as dependências necessárias:
    ```bash
    pip install -r requirements.txt
    ```

## Como usar

1. Inicie o servidor:
    ```bash
    python app.py
    ```

2. Acesse a interface web em `http://localhost:5000`.

## Estrutura do Projeto

- `app.py`: Arquivo principal que executa a aplicação.
- `app_utils.py`: Funções utilitárias usadas pela aplicação.
- `config.py`: Arquivo de configuração para parâmetros da câmera, modelos, etc.
- `model_cache.py`: Módulo para gerenciamento de cache de modelos.
- `models/`: Diretório onde os modelos de reconhecimento são armazenados.
- `templates/`: Arquivos HTML para a interface web.
- `static/`: Arquivos estáticos como CSS, JavaScript e imagens para a interface web.
- `requirements.txt`: Lista de dependências Python para instalar.

## Licença

Este projeto não possui uma licença definida.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou pull request.


## Autor
**Sandron Oliveira Silva**, Tecnólogo em Inteligência Artificial.
