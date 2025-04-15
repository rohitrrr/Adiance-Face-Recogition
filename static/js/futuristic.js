document.addEventListener('DOMContentLoaded', function() {
  // Particle background effect for hero sections
  const heroSections = document.querySelectorAll('.hero-section');
  
  heroSections.forEach(section => {
    particlesJS(section.id, {
      particles: {
        number: {
          value: 80,
          density: {
            enable: true,
            value_area: 800
          }
        },
        color: {
          value: "#ffffff"
        },
        shape: {
          type: "circle",
          stroke: {
            width: 0,
            color: "#000000"
          },
        },
        opacity: {
          value: 0.5,
          random: true,
          anim: {
            enable: true,
            speed: 1,
            opacity_min: 0.1,
            sync: false
          }
        },
        size: {
          value: 3,
          random: true,
          anim: {
            enable: true,
            speed: 2,
            size_min: 0.1,
            sync: false
          }
        },
        line_linked: {
          enable: true,
          distance: 150,
          color: "#ffffff",
          opacity: 0.4,
          width: 1
        },
        move: {
          enable: true,
          speed: 1,
          direction: "none",
          random: true,
          straight: false,
          out_mode: "out",
          bounce: false,
        }
      },
      interactivity: {
        detect_on: "canvas",
        events: {
          onhover: {
            enable: true,
            mode: "bubble"
          },
          onclick: {
            enable: true,
            mode: "push"
          },
          resize: true
        },
        modes: {
          bubble: {
            distance: 200,
            size: 6,
            duration: 2,
            opacity: 0.8,
            speed: 3
          },
          push: {
            particles_nb: 4
          }
        }
      },
      retina_detect: true
    });
  });

  // Enhanced file upload with preview
  const fileUploads = document.querySelectorAll('input[type="file"]');
  
  fileUploads.forEach(upload => {
    const container = upload.closest('.file-upload-container');
    if (!container) return;
    
    const preview = container.querySelector('.preview-container img');
    const dropZone = container.querySelector('.drop-zone');
    const label = container.querySelector('.file-name');
    
    // File selection handler
    upload.addEventListener('change', function() {
      if (this.files && this.files[0]) {
        const file = this.files[0];
        const reader = new FileReader();
        
        reader.onload = function(e) {
          if (preview) {
            preview.src = e.target.result;
            preview.style.display = 'block';
            
            // Fade in preview
            preview.style.opacity = 0;
            setTimeout(() => {
              preview.style.transition = 'opacity 0.5s ease';
              preview.style.opacity = 1;
            }, 50);
          }
          
          if (label) {
            label.textContent = file.name;
          }
        };
        
        reader.readAsDataURL(file);
      }
    });
    
    // Drag and drop handler
    if (dropZone) {
      ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, function(e) {
          e.preventDefault();
          dropZone.classList.add('active');
        });
      });
      
      ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, function(e) {
          e.preventDefault();
          dropZone.classList.remove('active');
          
          if (eventName === 'drop') {
            upload.files = e.dataTransfer.files;
            upload.dispatchEvent(new Event('change'));
          }
        });
      });
      
      // Click to open file dialog
      dropZone.addEventListener('click', () => {
        upload.click();
      });
    }
  });

  // Interactive charts for results
  const resultCharts = document.querySelectorAll('.result-chart');
  resultCharts.forEach(chart => {
    const ctx = chart.getContext('2d');
    const dataValues = JSON.parse(chart.dataset.values);
    const dataLabels = JSON.parse(chart.dataset.labels);
    const dataColors = JSON.parse(chart.dataset.colors);
    
    new Chart(ctx, {
      type: chart.dataset.type || 'bar',
      data: {
        labels: dataLabels,
        datasets: [{
          label: chart.dataset.label || 'Data',
          data: dataValues,
          backgroundColor: dataColors,
          borderColor: dataColors.map(color => color.replace('0.5', '1')),
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 2000,
          easing: 'easeOutQuart'
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              color: 'rgba(248, 250, 255, 0.7)'
            },
            grid: {
              color: 'rgba(248, 250, 255, 0.1)'
            }
          },
          x: {
            ticks: {
              color: 'rgba(248, 250, 255, 0.7)'
            },
            grid: {
              color: 'rgba(248, 250, 255, 0.1)'
            }
          }
        },
        plugins: {
          legend: {
            labels: {
              color: 'rgba(248, 250, 255, 0.9)'
            }
          }
        }
      }
    });
  });

  // Form submission loading states
  const forms = document.querySelectorAll('form:not(.no-loader)');
  forms.forEach(form => {
    form.addEventListener('submit', function() {
      const submitBtn = this.querySelector('[type="submit"]');
      if (submitBtn) {
        const originalText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        
        // Create loader
        const loader = document.createElement('div');
        loader.className = 'btn-loader';
        submitBtn.innerHTML = '';
        submitBtn.appendChild(loader);
        
        // Set timeout to prevent indefinite loading state
        setTimeout(() => {
          if (document.body.contains(submitBtn) && submitBtn.disabled) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
          }
        }, 20000);
      }
    });
  });

  // Scrolling animations
  const animateOnScroll = function() {
    const elements = document.querySelectorAll('.animate-on-scroll');
    
    elements.forEach(element => {
      const elementPosition = element.getBoundingClientRect().top;
      const screenPosition = window.innerHeight * 0.8;
      
      if (elementPosition < screenPosition) {
        element.classList.add('fade-in');
      }
    });
  };
  
  window.addEventListener('scroll', animateOnScroll);
  animateOnScroll(); // Run once on load
  
  // Glow effect on hover for specific elements
  const glowElements = document.querySelectorAll('.glow-on-hover');
  
  glowElements.forEach(element => {
    element.addEventListener('mousemove', function(e) {
      const rect = this.getBoundingClientRect();
      const x = e.clientX - rect.left; 
      const y = e.clientY - rect.top;
      
      this.style.background = `radial-gradient(circle at ${x}px ${y}px, rgba(var(--primary), 0.2), transparent)`;
    });
    
    element.addEventListener('mouseleave', function() {
      this.style.background = 'none';
    });
  });

  // Setup notification system
  window.showNotification = function(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type} animate-notification`;
    notification.innerHTML = `
      <div class="notification-icon">
        <i class="bi bi-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
      </div>
      <div class="notification-content">${message}</div>
      <button class="notification-close">&times;</button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-dismiss after 5 seconds
    const timeout = setTimeout(() => {
      notification.classList.add('notification-hide');
      setTimeout(() => notification.remove(), 300);
    }, 5000);
    
    // Manual dismiss
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.addEventListener('click', () => {
      clearTimeout(timeout);
      notification.classList.add('notification-hide');
      setTimeout(() => notification.remove(), 300);
    });
  };

  // Handle API calls with feedback
  const apiButtons = document.querySelectorAll('[data-api-action]');
  
  apiButtons.forEach(button => {
    button.addEventListener('click', async function(e) {
      e.preventDefault();
      
      const action = this.dataset.apiAction;
      const endpoint = this.dataset.apiEndpoint;
      const confirmMsg = this.dataset.apiConfirm;
      
      if (confirmMsg && !confirm(confirmMsg)) {
        return;
      }
      
      const originalText = this.innerHTML;
      this.disabled = true;
      this.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...`;
      
      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfToken()
          },
          body: JSON.stringify({ action })
        });
        
        const data = await response.json();
        
        if (data.success) {
          window.showNotification(data.message || 'Operation completed successfully', 'success');
          
          // Handle specific actions like reload or redirect
          if (data.redirect) {
            window.location.href = data.redirect;
            return;
          }
          
          if (data.reload) {
            window.location.reload();
            return;
          }
        } else {
          window.showNotification(data.message || 'Operation failed', 'error');
        }
      } catch (error) {
        window.showNotification('An unexpected error occurred', 'error');
        console.error('API Error:', error);
      } finally {
        this.disabled = false;
        this.innerHTML = originalText;
      }
    });
  });
  
  function getCsrfToken() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
  }
});

// Modern navigation menu for mobile
function toggleMobileMenu() {
  const menu = document.querySelector('.mobile-menu');
  menu.classList.toggle('open');
}

// Confetti effect for success actions
function launchConfetti() {
  confetti({
    particleCount: 100,
    spread: 70,
    origin: { y: 0.6 }
  });
}
