// Add loading state to button
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            const button = this.querySelector('button');
            button.classList.add('loading');
        });
    }
}); 