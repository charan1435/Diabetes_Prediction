// Main JavaScript for Diabetes Prediction App

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Initialize the application
function initializeApp() {
    setupFormValidation();
    setupTooltips();
    setupAnimations();
    setupPrintFunctionality();
    setupFormSubmission();
}

// Form Validation
function setupFormValidation() {
    const form = document.getElementById('predictionForm');
    if (!form) return;

    const inputs = form.querySelectorAll('input[type="number"]');
    
    inputs.forEach(input => {
        // Real-time validation
        input.addEventListener('input', function() {
            validateInput(this);
        });
        
        // Blur validation
        input.addEventListener('blur', function() {
            validateInput(this);
        });
    });

    // Form submission validation
    form.addEventListener('submit', function(e) {
        if (!validateForm(this)) {
            e.preventDefault();
            showAlert('Please correct the errors before submitting.', 'danger');
            return false;
        }
        
        // Show loading state
        showLoadingState(this);
    });
}

// Validate individual input
function validateInput(input) {
    const value = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const name = input.name;
    
    // Remove existing validation classes
    input.classList.remove('is-valid', 'is-invalid');
    
    // Remove existing feedback
    const existingFeedback = input.parentNode.querySelector('.invalid-feedback');
    if (existingFeedback) {
        existingFeedback.remove();
    }
    
    if (input.value === '') {
        return; // Don't validate empty fields
    }
    
    let isValid = true;
    let errorMessage = '';
    
    if (isNaN(value)) {
        isValid = false;
        errorMessage = 'Please enter a valid number.';
    } else if (value < min || value > max) {
        isValid = false;
        errorMessage = `Value must be between ${min} and ${max}.`;
    } else {
        // Additional specific validations
        switch(name) {
            case 'age':
                if (value < 1 || value > 120) {
                    isValid = false;
                    errorMessage = 'Age must be between 1 and 120 years.';
                }
                break;
            case 'bmi':
                if (value < 10 || value > 60) {
                    isValid = false;
                    errorMessage = 'BMI must be between 10 and 60 kg/m².';
                }
                break;
            case 'hba1c':
                if (value < 3 || value > 15) {
                    isValid = false;
                    errorMessage = 'HbA1c typically ranges from 3% to 15%.';
                }
                break;
        }
    }
    
    if (isValid) {
        input.classList.add('is-valid');
    } else {
        input.classList.add('is-invalid');
        
        // Add error feedback
        const feedback = document.createElement('div');
        feedback.className = 'invalid-feedback';
        feedback.textContent = errorMessage;
        input.parentNode.appendChild(feedback);
    }
    
    return isValid;
}

// Validate entire form
function validateForm(form) {
    const inputs = form.querySelectorAll('input[required], select[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (input.type === 'number') {
            if (!validateInput(input)) {
                isValid = false;
            }
        } else if (input.value.trim() === '') {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
            input.classList.add('is-valid');
        }
    });
    
    return isValid;
}

// Show loading state on form submission
function showLoadingState(form) {
    const submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn) {
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        submitBtn.disabled = true;
        
        // Store original text for potential restoration
        submitBtn.dataset.originalText = originalText;
    }
    
    // Add loading class to form
    form.classList.add('loading');
}

// Setup tooltips
function setupTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Setup animations
function setupAnimations() {
    // Fade in animation for cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.transition = 'all 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
    
    // Progress bar animations
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        
        setTimeout(() => {
            bar.style.transition = 'width 1s ease-in-out';
            bar.style.width = width;
        }, 500);
    });
}

// Setup print functionality
function setupPrintFunctionality() {
    // Add print button event listener
    const printBtns = document.querySelectorAll('[onclick*="print"]');
    printBtns.forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            printResults();
        });
    });
}

// Print results
function printResults() {
    // Hide elements that shouldn't be printed
    const noPrintElements = document.querySelectorAll('.no-print, .btn, nav, footer');
    noPrintElements.forEach(el => {
        el.style.display = 'none';
    });
    
    // Print the page
    window.print();
    
    // Restore hidden elements
    noPrintElements.forEach(el => {
        el.style.display = '';
    });
}

// Setup form submission with AJAX (optional)
function setupFormSubmission() {
    const form = document.getElementById('predictionForm');
    if (!form) return;
    
    // Add option for AJAX submission (commented out by default)
    /*
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        submitFormAjax(this);
    });
    */
}

// AJAX form submission (optional)
function submitFormAjax(form) {
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numeric fields
    ['age', 'urea', 'hba1c', 'chol', 'bmi', 'vldl'].forEach(field => {
        data[field] = parseFloat(data[field]);
    });
    
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            displayResults(data.predictions, data.input_data);
        } else {
            showAlert(data.error || 'Prediction failed', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Network error occurred', 'danger');
    })
    .finally(() => {
        hideLoadingState(form);
    });
}

// Display results (for AJAX)
function displayResults(predictions, inputData) {
    // This would dynamically update the page with results
    // Implementation depends on your UI requirements
    console.log('Predictions:', predictions);
    console.log('Input Data:', inputData);
}

// Hide loading state
function hideLoadingState(form) {
    const submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn && submitBtn.dataset.originalText) {
        submitBtn.innerHTML = submitBtn.dataset.originalText;
        submitBtn.disabled = false;
    }
    
    form.classList.remove('loading');
}

// Show alert message
function showAlert(message, type = 'info') {
    const alertContainer = document.querySelector('.container');
    if (!alertContainer) return;
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        <i class="fas fa-${type === 'danger' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the container
    alertContainer.insertBefore(alertDiv, alertContainer.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// BMI Calculator Helper
function calculateBMI(weight, height) {
    // Weight in kg, height in cm
    const heightM = height / 100;
    return (weight / (heightM * heightM)).toFixed(1);
}

// Input helpers
function addInputHelpers() {
    // BMI calculator popup (optional feature)
    const bmiInput = document.getElementById('bmi');
    if (bmiInput) {
        const helpBtn = document.createElement('button');
        helpBtn.type = 'button';
        helpBtn.className = 'btn btn-sm btn-outline-secondary';
        helpBtn.innerHTML = '<i class="fas fa-calculator"></i>';
        helpBtn.title = 'Calculate BMI';
        
        // Add after BMI input
        bmiInput.parentNode.appendChild(helpBtn);
        
        helpBtn.addEventListener('click', function() {
            showBMICalculator();
        });
    }
}

// BMI Calculator Modal (optional)
function showBMICalculator() {
    // Implementation for BMI calculator modal
    // This would show a popup to calculate BMI from height and weight
    alert('BMI Calculator feature - implement modal here');
}

// Utility functions
const utils = {
    // Format number with decimals
    formatNumber: (num, decimals = 1) => {
        return parseFloat(num).toFixed(decimals);
    },
    
    // Validate email
    validateEmail: (email) => {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    },
    
    // Get URL parameters
    getUrlParameter: (name) => {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get(name);
    },
    
    // Local storage helpers
    saveToLocal: (key, data) => {
        try {
            localStorage.setItem(key, JSON.stringify(data));
            return true;
        } catch (e) {
            console.error('Local storage error:', e);
            return false;
        }
    },
    
    loadFromLocal: (key) => {
        try {
            const data = localStorage.getItem(key);
            return data ? JSON.parse(data) : null;
        } catch (e) {
            console.error('Local storage error:', e);
            return null;
        }
    }
};

// Export utils for global access
window.DiabetesApp = {
    utils,
    validateInput,
    showAlert,
    calculateBMI
};

// Console welcome message
console.log('%c🩺 Diabetes Prediction System', 'color: #0d6efd; font-size: 16px; font-weight: bold;');
console.log('App initialized successfully!');