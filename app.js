// Smart Water AI Monitoring System - Frontend JavaScript

class WaterMonitoringApp {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDarkMode();
        this.updateDateTime();
        this.setupNotifications();
    }

    setupEventListeners() {
        // Usage form submission
        const usageForm = document.getElementById('usageForm');
        if (usageForm) {
            usageForm.addEventListener('submit', this.handleUsageSubmission.bind(this));
        }

        // Settings form
        const settingsForm = document.querySelector('form[method="POST"]');
        if (settingsForm && window.location.pathname === '/settings') {
            settingsForm.addEventListener('submit', this.handleSettingsUpdate.bind(this));
        }

        // Dark mode toggle
        const darkModeToggle = document.getElementById('dark_mode');
        if (darkModeToggle) {
            darkModeToggle.addEventListener('change', this.toggleDarkMode.bind(this));
        }
    }

    async handleUsageSubmission(event) {
        event.preventDefault();
        
        const submitButton = event.target.querySelector('button[type="submit"]');
        const originalText = submitButton.innerHTML;
        
        // Show loading state
        submitButton.innerHTML = '<span class="loading"></span> Adding...';
        submitButton.disabled = true;

        const formData = new FormData(event.target);
        const amount = parseFloat(formData.get('usageAmount') || document.getElementById('usageAmount').value);

        try {
            const response = await fetch('/api/usage', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ amount: amount })
            });

            const result = await response.json();

            if (result.success) {
                this.showNotification('Usage added successfully!', 'success');
                
                // Reset form
                event.target.reset();
                
                // Show alerts if any
                if (result.alerts && result.alerts.length > 0) {
                    result.alerts.forEach(alert => {
                        this.showNotification(alert.message, 'warning');
                    });
                }
                
                // Update dashboard data
                this.updateDashboardData();
            } else {
                throw new Error('Failed to add usage');
            }
        } catch (error) {
            console.error('Error:', error);
            this.showNotification('Error adding usage. Please try again.', 'danger');
        } finally {
            // Restore button state
            submitButton.innerHTML = originalText;
            submitButton.disabled = false;
        }
    }

    handleSettingsUpdate(event) {
        const submitButton = event.target.querySelector('button[type="submit"]');
        const originalText = submitButton.innerHTML;
        
        // Show loading state
        submitButton.innerHTML = '<span class="loading"></span> Saving...';
        submitButton.disabled = true;

        // The form will submit normally, but we show loading state
        setTimeout(() => {
            submitButton.innerHTML = originalText;
            submitButton.disabled = false;
        }, 2000);
    }

    async updateDashboardData() {
        try {
            const response = await fetch('/api/usage-data');
            const data = await response.json();
            
            // Update today's usage
            const today = new Date().toISOString().split('T')[0];
            const todayData = data.find(item => item[0] === today);
            const todayUsage = todayData ? todayData[1] : 0;
            
            const todayUsageElement = document.getElementById('todayUsage');
            if (todayUsageElement) {
                todayUsageElement.textContent = `${todayUsage.toFixed(1)}L`;
            }
            
            // Update chart if it exists
            if (window.chart) {
                window.chart.data.labels = data.map(item => item[0]);
                window.chart.data.datasets[0].data = data.map(item => item[1]);
                window.chart.update();
            }
        } catch (error) {
            console.error('Error updating dashboard data:', error);
        }
    }

    showNotification(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    setupDarkMode() {
        const darkModeToggle = document.getElementById('dark_mode');
        if (darkModeToggle) {
            // Check for saved dark mode preference
            const isDarkMode = localStorage.getItem('darkMode') === 'true';
            darkModeToggle.checked = isDarkMode;
            this.applyDarkMode(isDarkMode);
        }
    }

    toggleDarkMode(event) {
        const isDarkMode = event.target.checked;
        localStorage.setItem('darkMode', isDarkMode);
        this.applyDarkMode(isDarkMode);
    }

    applyDarkMode(isDarkMode) {
        if (isDarkMode) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
    }

    updateDateTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        const dateString = now.toLocaleDateString();
        
        // Update any datetime displays
        const datetimeElements = document.querySelectorAll('.current-datetime');
        datetimeElements.forEach(element => {
            element.textContent = `${dateString} ${timeString}`;
        });
        
        // Update every minute
        setTimeout(() => this.updateDateTime(), 60000);
    }

    setupNotifications() {
        // Request notification permission
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
    }

    showBrowserNotification(title, message) {
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification(title, {
                body: message,
                icon: '/static/images/water-icon.png',
                badge: '/static/images/water-icon.png'
            });
        }
    }

    // Utility function to format numbers
    formatNumber(num, decimals = 1) {
        return parseFloat(num).toFixed(decimals);
    }

    // Utility function to format dates
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    }

    // Check for usage alerts periodically
    startAlertMonitoring() {
        setInterval(async () => {
            try {
                const response = await fetch('/api/check-alerts');
                const alerts = await response.json();
                
                alerts.forEach(alert => {
                    this.showNotification(alert.message, 'warning');
                    this.showBrowserNotification('Water Usage Alert', alert.message);
                });
            } catch (error) {
                console.error('Error checking alerts:', error);
            }
        }, 300000); // Check every 5 minutes
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new WaterMonitoringApp();
    
    // Start alert monitoring if on dashboard
    if (window.location.pathname === '/dashboard') {
        app.startAlertMonitoring();
    }
});

// Service Worker for offline functionality (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

// Export for use in other scripts
window.WaterMonitoringApp = WaterMonitoringApp;