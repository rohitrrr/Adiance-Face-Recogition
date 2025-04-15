document.addEventListener('DOMContentLoaded', function() {
    // Image preview functionality
    const fileInput = document.getElementById('file');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.querySelector('.preview-container');
    
    if (fileInput && imagePreview) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                    previewContainer.classList.add('fade-in');
                    
                    // Reset animation
                    setTimeout(() => {
                        previewContainer.classList.remove('fade-in');
                    }, 500);
                }
                
                reader.readAsDataURL(this.files[0]);
            }
        });
    }
    
    // Form submission handling with loading indicators
    const forms = document.querySelectorAll('form:not(.no-loader)');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('[type="submit"]');
            if (submitBtn) {
                const originalText = submitBtn.innerHTML;
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                
                // Revert back after timeout if form is still visible (for slow submissions)
                setTimeout(() => {
                    if (document.body.contains(submitBtn) && submitBtn.disabled) {
                        submitBtn.disabled = false;
                        submitBtn.innerHTML = originalText;
                    }
                }, 15000);
            }
        });
    });
    
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert-dismissible:not(.no-auto-dismiss)');
    alerts.forEach(alert => {
        setTimeout(() => {
            const closeBtn = alert.querySelector('.btn-close');
            if (closeBtn) {
                closeBtn.click();
            }
        }, 5000);
    });
    
    // Setup button handling
    const setupButton = document.getElementById('setupButton');
    if (setupButton) {
        setupButton.addEventListener('click', function(e) {
            e.preventDefault();
            this.disabled = true;
            this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Setting up...';
            
            fetch('/setup')
                .then(response => response.json())
                .then(data => {
                    const alertClass = data.status === 'success' ? 'alert-success' : 'alert-danger';
                    const alertDiv = document.createElement('div');
                    alertDiv.className = `alert ${alertClass} alert-dismissible fade show`;
                    alertDiv.innerHTML = `
                        ${data.message}
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    `;
                    document.querySelector('.alerts-container').appendChild(alertDiv);
                    
                    this.disabled = false;
                    this.innerHTML = 'Setup Complete';
                    
                    setTimeout(() => {
                        const closeBtn = alertDiv.querySelector('.btn-close');
                        if (closeBtn) closeBtn.click();
                    }, 5000);
                })
                .catch(error => {
                    console.error('Error:', error);
                    this.disabled = false;
                    this.innerHTML = 'Retry Setup';
                });
        });
    }
});
