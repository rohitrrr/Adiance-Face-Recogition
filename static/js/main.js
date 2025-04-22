/**
 * Advanced Futuristic UI Animations
 * Adiance Face Recognition System
 */

// Wait for DOM content to fully load
document.addEventListener('DOMContentLoaded', function() {
    // Page loading animation
    const pageLoader = document.getElementById('page-loader');
    if (pageLoader) {
        setTimeout(() => {
            pageLoader.classList.add('loaded');
            setTimeout(() => {
                pageLoader.style.display = 'none';
            }, 600);
        }, 1000);
    }

    // Initialize particle backgrounds for hero sections
    initParticleBackgrounds();
    
    // Add hover effects to UI elements
    addHolographicEffects();
    
    // Initialize glitch effects for text
    initGlitchEffects();
    
    // Setup animated counters
    initCounters();
    
    // Initialize scanner animations
    initScannerEffects();
});

/**
 * Initialize particle background effects
 */
function initParticleBackgrounds() {
    const particleSections = document.querySelectorAll('.hero-section, .particle-bg');
    
    particleSections.forEach(section => {
        const id = section.id || `particle-bg-${Math.random().toString(36).substring(2, 9)}`;
        section.id = id;
        
        const particleDiv = document.createElement('div');
        particleDiv.className = 'particle-container';
        section.prepend(particleDiv);
        
        particlesJS(id, {
            "particles": {
                "number": {
                    "value": 80,
                    "density": {"enable": true, "value_area": 800}
                },
                "color": {"value": "#ffffff"},
                "shape": {
                    "type": "circle",
                    "stroke": {"width": 0, "color": "#000000"},
                    "polygon": {"nb_sides": 5}
                },
                "opacity": {
                    "value": 0.3,
                    "random": true,
                    "anim": {"enable": true, "speed": 1, "opacity_min": 0.1, "sync": false}
                },
                "size": {
                    "value": 3,
                    "random": true,
                    "anim": {"enable": true, "speed": 2, "size_min": 0.1, "sync": false}
                },
                "line_linked": {
                    "enable": true,
                    "distance": 150,
                    "color": "#00ccff",
                    "opacity": 0.2,
                    "width": 1
                },
                "move": {
                    "enable": true,
                    "speed": 1,
                    "direction": "none",
                    "random": true,
                    "straight": false,
                    "out_mode": "out",
                    "bounce": false,
                    "attract": {"enable": false, "rotateX": 600, "rotateY": 1200}
                }
            },
            "interactivity": {
                "detect_on": "canvas",
                "events": {
                    "onhover": {"enable": true, "mode": "grab"},
                    "onclick": {"enable": true, "mode": "push"},
                    "resize": true
                },
                "modes": {
                    "grab": {"distance": 150, "line_linked": {"opacity": 0.3}},
                    "push": {"particles_nb": 4},
                    "remove": {"particles_nb": 2}
                }
            },
            "retina_detect": true
        });
    });
}

/**
 * Add holographic hover effects to cards and UI elements
 */
function addHolographicEffects() {
    const cards = document.querySelectorAll('.futuristic-card, .glow-on-hover');
    
    cards.forEach(card => {
        // Holographic effect on mouse move
        card.addEventListener('mousemove', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Calculate position percentage
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const percentX = (x - centerX) / centerX;
            const percentY = (y - centerY) / centerY;
            
            // Apply 3D rotation (subtle)
            this.style.transform = `perspective(1000px) rotateY(${percentX * 3}deg) rotateX(${-percentY * 3}deg)`;
            
            // Dynamic gradient overlay based on mouse position
            this.style.background = `radial-gradient(circle at ${x}px ${y}px, 
                rgba(var(--primary), 0.15), 
                rgba(var(--accent), 0.05) 40%, 
                transparent 60%)`;
                
            // Add holographic reflection effect
            if (!this.querySelector('.holographic-reflection')) {
                const reflection = document.createElement('div');
                reflection.className = 'holographic-reflection';
                this.appendChild(reflection);
            }
            
            const reflection = this.querySelector('.holographic-reflection');
            reflection.style.opacity = '0.2';
            reflection.style.left = `${x / rect.width * 100}%`;
            reflection.style.top = `${y / rect.height * 100}%`;
        });
        
        // Reset on mouse leave
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'perspective(1000px) rotateY(0) rotateX(0)';
            this.style.background = '';
            
            const reflection = this.querySelector('.holographic-reflection');
            if (reflection) {
                reflection.style.opacity = '0';
            }
        });
        
        // Add transition for smooth effect
        card.style.transition = 'transform 0.2s ease, background 0.3s ease';
    });
    
    // Enhanced menu hover effects
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('mouseenter', function() {
            this.classList.add('nav-hover');
        });
        
        link.addEventListener('mouseleave', function() {
            this.classList.remove('nav-hover');
        });
    });
}

/**
 * Initialize glitch text effects on specified elements
 */
function initGlitchEffects() {
    const glitchElements = document.querySelectorAll('.glitch-text');
    
    glitchElements.forEach(element => {
        const text = element.textContent;
        element.setAttribute('data-text', text);
        
        // Random glitch animation
        setInterval(() => {
            if (Math.random() > 0.95) {
                element.classList.add('glitch-active');
                setTimeout(() => {
                    element.classList.remove('glitch-active');
                }, 200);
            }
        }, 2000);
    });
}

/**
 * Initialize animated counters for stats
 */
function initCounters() {
    const statValues = document.querySelectorAll('.stat-card .value');
    
    statValues.forEach(value => {
        const targetValue = parseFloat(value.textContent);
        const duration = 2000; // ms
        const startTime = Date.now();
        
        if (isNaN(targetValue)) return;
        
        // Start from zero
        value.textContent = '0';
        
        // Set current suffix if present (like % or ms)
        const suffix = value.innerHTML.replace(/[0-9.]/g, '');
        
        // Animate the counter
        const counterAnimation = () => {
            const now = Date.now();
            const progress = Math.min((now - startTime) / duration, 1);
            const currentValue = progress * targetValue;
            
            // Format the value appropriately (integers or decimals)
            value.textContent = Number.isInteger(targetValue) ? 
                Math.floor(currentValue) + suffix : 
                currentValue.toFixed(1) + suffix;
            
            if (progress < 1) {
                requestAnimationFrame(counterAnimation);
            }
        };
        
        requestAnimationFrame(counterAnimation);
    });
}

/**
 * Initialize scanner animations for search and enrollment pages
 */
function initScannerEffects() {
    // Advanced scanner animation for preview containers
    const previewContainers = document.querySelectorAll('.preview-container');
    
    previewContainers.forEach(container => {
        // Enhance existing scan line with additional effects
        let scanLine = container.querySelector('.scan-line');
        if (!scanLine) {
            scanLine = document.createElement('div');
            scanLine.className = 'scan-line';
            container.appendChild(scanLine);
        }
        
        // Add data grid overlay for futuristic effect
        const dataGrid = document.createElement('div');
        dataGrid.className = 'data-grid';
        container.appendChild(dataGrid);
        
        // Create grid pattern
        for (let i = 0; i < 10; i++) {
            const horizontalLine = document.createElement('div');
            horizontalLine.className = 'grid-line horizontal';
            horizontalLine.style.top = `${i * 10}%`;
            dataGrid.appendChild(horizontalLine);
            
            const verticalLine = document.createElement('div');
            verticalLine.className = 'grid-line vertical';
            verticalLine.style.left = `${i * 10}%`;
            dataGrid.appendChild(verticalLine);
        }
        
        // Add scan effect on hover
        container.addEventListener('mouseenter', function() {
            scanLine.style.opacity = '1';
            scanLine.style.animation = 'scanLine 2s ease-in-out infinite';
            dataGrid.classList.add('active');
        });
        
        container.addEventListener('mouseleave', function() {
            scanLine.style.opacity = '0';
            dataGrid.classList.remove('active');
        });
    });
}

// Handle dynamic UI interactions
window.addEventListener('scroll', function() {
    // Add parallax effect to blobs and hero elements
    document.querySelectorAll('.blob').forEach(blob => {
        const scrollY = window.scrollY;
        const speed = parseFloat(blob.dataset.speed || 0.05);
        blob.style.transform = `translateY(${scrollY * speed}px)`;
    });
    
    // Add floating animation to feature icons when scrolled into view
    document.querySelectorAll('.feature-icon').forEach(icon => {
        if (isElementInViewport(icon) && !icon.classList.contains('floating')) {
            icon.classList.add('floating');
        }
    });
});

// Helper function to check if element is in viewport
function isElementInViewport(el) {
    const rect = el.getBoundingClientRect();
    return (
        rect.top <= (window.innerHeight || document.documentElement.clientHeight) * 0.9 &&
        rect.bottom >= 0
    );
}
