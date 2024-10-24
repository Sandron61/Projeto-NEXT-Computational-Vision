// Arquivo JavaScript personalizado

// Exemplo: Função para exibir uma mensagem de boas-vindas no console
document.addEventListener('DOMContentLoaded', function() {
    console.log('Aplicação carregada com sucesso.');
});

// Fechar mensagens flash
document.querySelectorAll('.flash-message .close-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
        this.parentElement.style.display = 'none';
    });
});

// Smooth Scroll para links internos (se houver)
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if(target){
            target.scrollIntoView({
                behavior: 'smooth'
            });
        }
    });
});
// Iniciar o áudio de celebração quando a página de pódio for carregada
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('celebration-music')) {
        const celebrationMusic = document.getElementById('celebration-music');
        celebrationMusic.volume = 0.5; // Ajuste o volume conforme necessário
    }
});