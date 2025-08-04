from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import sqlite3
import hashlib
import os
import json
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import io
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Simple session management
from flask import session

# Create static directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

def init_db():
    conn = sqlite3.connect('water_monitoring.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            daily_limit REAL DEFAULT 100,
            hourly_limit REAL DEFAULT 10
        )
    ''')
    
    # Water usage table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS water_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            usage_amount REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            alert_type TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_read BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # ML Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prediction_date DATE,
            predicted_usage REAL,
            actual_usage REAL DEFAULT NULL,
            model_accuracy REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

def export_user_data_to_csv(user_id):
    """Export user's water usage data to CSV for ML processing"""
    conn = sqlite3.connect('water_monitoring.db')
    
    # Daily usage data
    daily_query = '''
        SELECT DATE(timestamp) as date, SUM(usage_amount) as total_usage
        FROM water_usage 
        WHERE user_id = ?
        GROUP BY DATE(timestamp)
        ORDER BY date
    '''
    daily_df = pd.read_sql_query(daily_query, conn, params=(user_id,))
    daily_df['day_number'] = range(1, len(daily_df) + 1)
    
    # Hourly usage data for today
    today = datetime.now().date()
    hourly_query = '''
        SELECT strftime('%H', timestamp) as hour, SUM(usage_amount) as total_usage
        FROM water_usage 
        WHERE user_id = ? AND DATE(timestamp) = ?
        GROUP BY strftime('%H', timestamp)
        ORDER BY hour
    '''
    hourly_df = pd.read_sql_query(hourly_query, conn, params=(user_id, today))
    
    conn.close()
    
    # Save to CSV files
    daily_csv_path = f'static/user_{user_id}_daily_usage.csv'
    hourly_csv_path = f'static/user_{user_id}_hourly_usage.csv'
    
    if not daily_df.empty:
        daily_df.to_csv(daily_csv_path, index=False)
    
    if not hourly_df.empty:
        hourly_df.to_csv(hourly_csv_path, index=False)
    
    return daily_df, hourly_df, daily_csv_path, hourly_csv_path

def generate_ml_predictions(user_id):
    """Generate ML-based water usage predictions"""
    try:
        daily_df, hourly_df, daily_csv, hourly_csv = export_user_data_to_csv(user_id)
        
        if len(daily_df) < 3:
            return None, "Insufficient data for predictions (need at least 3 days)"
        
        # Prepare data for ML model
        X = daily_df[['day_number']].values
        y = daily_df['total_usage'].values
        
        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate model accuracy
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Predict next 5 days
        last_day = daily_df['day_number'].max()
        future_days = np.array([[last_day + i] for i in range(1, 6)])
        future_predictions = model.predict(future_days)
        
        # Create prediction data
        predictions = []
        for i, pred in enumerate(future_predictions):
            pred_date = datetime.now().date() + timedelta(days=i+1)
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'predicted_usage': round(max(0, pred), 1),  # Ensure non-negative
                'day_number': last_day + i + 1
            })
        
        # Generate prediction chart
        chart_path = generate_prediction_chart(daily_df, predictions, user_id)
        
        # Save predictions to database
        save_predictions_to_db(user_id, predictions, r2)
        
        return {
            'predictions': predictions,
            'model_accuracy': round(r2 * 100, 1),
            'mean_error': round(mae, 1),
            'chart_path': chart_path,
            'trend': 'increasing' if model.coef_[0] > 0 else 'decreasing',
            'daily_change': round(model.coef_[0], 2)
        }, None
        
    except Exception as e:
        print(f"ML Prediction Error: {e}")
        return None, str(e)

def generate_prediction_chart(daily_df, predictions, user_id):
    """Generate matplotlib chart for predictions"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        dates = pd.to_datetime(daily_df['date'])
        plt.plot(dates, daily_df['total_usage'], 
                marker='o', linewidth=2, markersize=6, 
                label='Actual Usage', color='#0d6efd')
        
        # Plot predictions
        pred_dates = [datetime.strptime(p['date'], '%Y-%m-%d') for p in predictions]
        pred_values = [p['predicted_usage'] for p in predictions]
        plt.plot(pred_dates, pred_values, 
                marker='x', linewidth=2, markersize=8, 
                linestyle='--', label='Predicted Usage', color='#dc3545')
        
        # Formatting
        plt.title('Water Usage Forecast - Next 5 Days', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Water Usage (Liters)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        
        plt.tight_layout()
        
        # Save chart
        chart_path = f'static/prediction_chart_user_{user_id}.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Chart generation error: {e}")
        return None

def save_predictions_to_db(user_id, predictions, accuracy):
    """Save ML predictions to database"""
    conn = sqlite3.connect('water_monitoring.db')
    cursor = conn.cursor()
    
    for pred in predictions:
        cursor.execute('''
            INSERT OR REPLACE INTO ml_predictions 
            (user_id, prediction_date, predicted_usage, model_accuracy)
            VALUES (?, ?, ?, ?)
        ''', (user_id, pred['date'], pred['predicted_usage'], accuracy))
    
    conn.commit()
    conn.close()

def detect_usage_anomalies(user_id):
    """Detect unusual usage patterns and spikes"""
    conn = sqlite3.connect('water_monitoring.db')
    cursor = conn.cursor()
    
    # Get recent usage data
    cursor.execute('''
        SELECT DATE(timestamp) as date, SUM(usage_amount) as daily_usage
        FROM water_usage 
        WHERE user_id = ? AND timestamp >= date('now', '-14 days')
        GROUP BY DATE(timestamp)
        ORDER BY date
    ''', (user_id,))
    
    daily_data = cursor.fetchall()
    
    if len(daily_data) < 7:
        conn.close()
        return []
    
    # Calculate statistics
    usage_values = [float(row[1]) for row in daily_data]
    avg_usage = np.mean(usage_values)
    std_usage = np.std(usage_values)
    
    anomalies = []
    
    # Detect daily anomalies (usage > 2 standard deviations from mean)
    for date, usage in daily_data[-7:]:  # Last 7 days
        if float(usage) > avg_usage + (2 * std_usage):
            anomalies.append({
                'type': 'Daily Spike',
                'date': date,
                'usage': float(usage),
                'message': f'🚨 Unusual high usage: {usage:.1f}L (avg: {avg_usage:.1f}L)',
                'severity': 'high'
            })
        elif float(usage) > avg_usage + (1.5 * std_usage):
            anomalies.append({
                'type': 'Daily Warning',
                'date': date,
                'usage': float(usage),
                'message': f'⚠️ Above normal usage: {usage:.1f}L (avg: {avg_usage:.1f}L)',
                'severity': 'medium'
            })
    
    # Detect hourly anomalies for today
    today = datetime.now().date()
    cursor.execute('''
        SELECT strftime('%H', timestamp) as hour, SUM(usage_amount) as hourly_usage
        FROM water_usage 
        WHERE user_id = ? AND DATE(timestamp) = ?
        GROUP BY strftime('%H', timestamp)
        ORDER BY hour
    ''', (user_id, today))
    
    hourly_data = cursor.fetchall()
    
    if hourly_data:
        hourly_values = [float(row[1]) for row in hourly_data]
        avg_hourly = np.mean(hourly_values)
        
        for hour, usage in hourly_data:
            if float(usage) > 1.8 * avg_hourly and float(usage) > 5:  # Significant spike
                anomalies.append({
                    'type': 'Hourly Spike',
                    'date': str(today),
                    'hour': hour,
                    'usage': float(usage),
                    'message': f'⏰ Hourly spike at {hour}:00 - {usage:.1f}L',
                    'severity': 'high'
                })
    
    conn.close()
    return anomalies

def add_demo_data(user_id):
    """Add comprehensive demo water usage data for the past 14 days"""
    conn = sqlite3.connect('water_monitoring.db')
    cursor = conn.cursor()
    
    # Check if demo data already exists
    cursor.execute('SELECT COUNT(*) FROM water_usage WHERE user_id = ?', (user_id,))
    existing_count = cursor.fetchone()[0]
    
    if existing_count > 20:  # Allow some real usage data
        print(f"Demo data already exists for user {user_id}")
        conn.close()
        return
    
    print(f"Adding comprehensive demo data for user {user_id}...")
    
    # Clear existing demo data to avoid duplicates
    cursor.execute('DELETE FROM water_usage WHERE user_id = ?', (user_id,))
    cursor.execute('DELETE FROM alerts WHERE user_id = ?', (user_id,))
    
    # Generate realistic water usage patterns for past 14 days
    base_usage = 85
    trend_increase = 2  # Gradual increase over time
    
    for day_offset in range(14, 0, -1):
        target_date = datetime.now() - timedelta(days=day_offset)
        
        # Calculate daily target with trend and weekly pattern
        daily_trend = trend_increase * (14 - day_offset) / 14
        weekend_multiplier = 1.3 if target_date.weekday() >= 5 else 1.0
        random_variation = random.uniform(0.8, 1.2)
        
        daily_target = (base_usage + daily_trend) * weekend_multiplier * random_variation
        
        # Add some anomalies for testing
        if day_offset in [3, 8]:  # Create spikes on specific days
            daily_target *= 1.6
        
        # Peak hours for realistic distribution
        peak_hours = [7, 8, 12, 18, 19, 21]
        
        daily_total = 0
        for hour in range(24):
            if hour in peak_hours:
                hourly_usage = daily_target * 0.7 / len(peak_hours)
                # Add hourly spikes occasionally
                if day_offset == 1 and hour == 19:  # Today evening spike
                    hourly_usage *= 2.5
            else:
                hourly_usage = daily_target * 0.3 / (24 - len(peak_hours))
            
            # Add some randomness
            hourly_usage *= random.uniform(0.5, 1.5)
            hourly_usage = max(0.1, hourly_usage)
            
            # Create multiple entries per hour
            entries = random.randint(1, 4) if hour in peak_hours else random.randint(0, 2)
            
            if entries > 0:
                usage_per_entry = hourly_usage / entries
                for entry in range(entries):
                    minute = random.randint(0, 59)
                    second = random.randint(0, 59)
                    usage_time = target_date.replace(hour=hour, minute=minute, second=second)
                    
                    cursor.execute('''
                        INSERT INTO water_usage (user_id, usage_amount, timestamp)
                        VALUES (?, ?, ?)
                    ''', (user_id, round(usage_per_entry, 1), usage_time))
                    
                    daily_total += usage_per_entry
        
        print(f"Day {day_offset}: Generated {daily_total:.1f}L")
    
    conn.commit()
    conn.close()
    print("✅ Enhanced demo data with trends and anomalies added!")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_current_user():
    if 'user_id' in session:
        conn = sqlite3.connect('water_monitoring.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],))
        user_data = cursor.fetchone()
        conn.close()
        if user_data:
            return {
                'id': user_data[0],
                'email': user_data[1],
                'name': user_data[2],
                'daily_limit': user_data[4],
                'hourly_limit': user_data[5]
            }
    return None

def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def get_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning"
    elif 12 <= hour < 17:
        return "Good Afternoon"
    elif 17 <= hour < 21:
        return "Good Evening"
    else:
        return "Good Night"

def get_water_usage_data(user_id, days=7):
    """Get daily water usage data for the past N days"""
    conn = sqlite3.connect('water_monitoring.db')
    cursor = conn.cursor()
    
    # Get data for each of the past N days
    usage_data = []
    
    for day_offset in range(days, 0, -1):
        target_date = datetime.now() - timedelta(days=day_offset)
        date_str = target_date.strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT COALESCE(SUM(usage_amount), 0) as total_usage
            FROM water_usage 
            WHERE user_id = ? AND DATE(timestamp) = ?
        ''', (user_id, date_str))
        
        result = cursor.fetchone()
        total_usage = float(result[0]) if result else 0
        
        # Format for chart display
        display_date = target_date.strftime('%m/%d')
        usage_data.append([display_date, total_usage])
    
    conn.close()
    return usage_data

def get_hourly_usage_data(user_id):
    """Get hourly usage data for today"""
    conn = sqlite3.connect('water_monitoring.db')
    cursor = conn.cursor()
    
    today = datetime.now().date()
    cursor.execute('''
        SELECT strftime('%H', timestamp) as hour, SUM(usage_amount) as total_usage
        FROM water_usage 
        WHERE user_id = ? AND DATE(timestamp) = ?
        GROUP BY strftime('%H', timestamp)
        ORDER BY hour
    ''', (user_id, today))
    
    data = cursor.fetchall()
    conn.close()
    
    # Create 24-hour array with 0 for missing hours
    hourly_data = [0] * 24
    for row in data:
        hour = int(row[0])
        usage = float(row[1])
        hourly_data[hour] = usage
    
    return hourly_data

def get_current_usage_stats(user_id):
    conn = sqlite3.connect('water_monitoring.db')
    cursor = conn.cursor()
    
    # Current hour usage
    hour_ago = datetime.now() - timedelta(hours=1)
    cursor.execute('''
        SELECT COALESCE(SUM(usage_amount), 0) FROM water_usage 
        WHERE user_id = ? AND timestamp >= ?
    ''', (user_id, hour_ago))
    hourly_usage = cursor.fetchone()[0]
    
    # Current day usage
    today = datetime.now().date()
    cursor.execute('''
        SELECT COALESCE(SUM(usage_amount), 0) FROM water_usage 
        WHERE user_id = ? AND DATE(timestamp) = ?
    ''', (user_id, today))
    daily_usage = cursor.fetchone()[0]
    
    # Weekly usage
    week_ago = datetime.now() - timedelta(days=7)
    cursor.execute('''
        SELECT COALESCE(SUM(usage_amount), 0) FROM water_usage 
        WHERE user_id = ? AND timestamp >= ?
    ''', (user_id, week_ago))
    weekly_usage = cursor.fetchone()[0]
    
    conn.close()
    return float(hourly_usage), float(daily_usage), float(weekly_usage)

def check_usage_alerts(user_id, usage_amount):
    user = get_current_user()
    if not user:
        return []
    
    alerts = []
    conn = sqlite3.connect('water_monitoring.db')
    cursor = conn.cursor()
    
    # Get current usage stats
    hourly_usage, daily_usage, _ = get_current_usage_stats(user_id)
    hourly_usage += usage_amount
    daily_usage += usage_amount
    
    # Hourly limit check
    if hourly_usage > user['hourly_limit']:
        alerts.append({
            'type': 'hourly_exceeded',
            'message': f'🚨 Hourly limit exceeded! Used {hourly_usage:.1f}L (Limit: {user["hourly_limit"]}L)',
            'severity': 'high'
        })
    elif hourly_usage > user['hourly_limit'] * 0.8:
        alerts.append({
            'type': 'hourly_warning',
            'message': f'⚠️ Approaching hourly limit! Used {hourly_usage:.1f}L (Limit: {user["hourly_limit"]}L)',
            'severity': 'medium'
        })
    
    # Daily limit check
    if daily_usage > user['daily_limit']:
        alerts.append({
            'type': 'daily_exceeded',
            'message': f'🚨 Daily limit exceeded! Used {daily_usage:.1f}L (Limit: {user["daily_limit"]}L)',
            'severity': 'high'
        })
    elif daily_usage > user['daily_limit'] * 0.8:
        alerts.append({
            'type': 'daily_warning',
            'message': f'⚠️ Approaching daily limit! Used {daily_usage:.1f}L (Limit: {user["daily_limit"]}L)',
            'severity': 'medium'
        })
    
    # Save alerts to database
    for alert in alerts:
        cursor.execute('''
            INSERT INTO alerts (user_id, alert_type, message)
            VALUES (?, ?, ?)
        ''', (user_id, alert['type'], alert['message']))
    
    conn.commit()
    conn.close()
    
    return alerts

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Water AI Monitoring</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .card { border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
            .feature-icon { font-size: 2rem; color: #667eea; margin-bottom: 1rem; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row justify-content-center align-items-center min-vh-100">
                <div class="col-md-10">
                    <div class="card">
                        <div class="card-body text-center p-5">
                            <i class="fas fa-tint fa-5x text-primary mb-4"></i>
                            <h1 class="display-4 text-primary">Smart Water AI</h1>
                            <p class="lead text-muted mb-4">Advanced water monitoring with ML predictions and anomaly detection</p>
                            
                            <!-- Features -->
                            <div class="row mb-4">
                                <div class="col-md-3">
                                    <i class="fas fa-chart-line feature-icon"></i>
                                    <h6>Real-time Monitoring</h6>
                                    <small class="text-muted">Track usage patterns</small>
                                </div>
                                <div class="col-md-3">
                                    <i class="fas fa-brain feature-icon"></i>
                                    <h6>ML Predictions</h6>
                                    <small class="text-muted">5-day usage forecasts</small>
                                </div>
                                <div class="col-md-3">
                                    <i class="fas fa-exclamation-triangle feature-icon"></i>
                                    <h6>Anomaly Detection</h6>
                                    <small class="text-muted">Detect unusual spikes</small>
                                </div>
                                <div class="col-md-3">
                                    <i class="fas fa-lightbulb feature-icon"></i>
                                    <h6>AI Insights</h6>
                                    <small class="text-muted">Smart recommendations</small>
                                </div>
                            </div>
                            
                            <div class="d-grid gap-2 d-md-flex justify-content-md-center">
                                <a href="/login" class="btn btn-primary btn-lg me-md-2">
                                    <i class="fas fa-sign-in-alt"></i> Sign In
                                </a>
                                <a href="/register" class="btn btn-outline-primary btn-lg">
                                    <i class="fas fa-user-plus"></i> Sign Up
                                </a>
                            </div>
                            
                            <div class="mt-4">
                                <small class="text-muted">
                                    <i class="fas fa-info-circle"></i> 
                                    Demo includes 14 days of data with ML predictions and anomaly detection
                                </small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        password = request.form['password']
        
        print(f"Registration attempt: {email}, {name}")
        
        conn = sqlite3.connect('water_monitoring.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        
        if cursor.fetchone():
            conn.close()
            return '''
            <script>
                alert('Email already registered!');
                window.location.href = '/register';
            </script>
            '''
        
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (email, name, password_hash)
            VALUES (?, ?, ?)
        ''', (email, name, password_hash))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Add comprehensive demo data for new user
        add_demo_data(user_id)
        
        print(f"User registered successfully: {email}")
        
        return '''
        <script>
            alert('Registration successful! ML-ready demo data added. Please login.');
            window.location.href = '/login';
        </script>
        '''
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Register - Smart Water AI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .card { border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row justify-content-center align-items-center min-vh-100">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body p-5">
                            <div class="text-center mb-4">
                                <i class="fas fa-tint fa-3x text-primary mb-3"></i>
                                <h3>Create Account</h3>
                                <small class="text-muted">14 days of ML-ready demo data will be added</small>
                            </div>
                            <form method="POST">
                                <div class="mb-3">
                                    <label class="form-label">Full Name</label>
                                    <input type="text" class="form-control" name="name" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Email</label>
                                    <input type="email" class="form-control" name="email" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Password</label>
                                    <input type="password" class="form-control" name="password" required>
                                </div>
                                <div class="d-grid">
                                    <button type="submit" class="btn btn-primary">
                                        <i class="fas fa-user-plus"></i> Sign Up
                                    </button>
                                </div>
                            </form>
                            <div class="text-center mt-3">
                                <a href="/login">Already have an account? Sign in</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        print(f"Login attempt: {email}")
        
        conn = sqlite3.connect('water_monitoring.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user_data = cursor.fetchone()
        conn.close()
        
        if user_data:
            stored_hash = user_data[3]
            input_hash = hash_password(password)
            
            if stored_hash == input_hash:
                session['user_id'] = user_data[0]
                print(f"Login successful for user: {user_data[1]}")
                
                # Add demo data if user doesn't have sufficient data
                add_demo_data(user_data[0])
                
                return redirect(url_for('dashboard'))
            else:
                print("Password mismatch")
        else:
            print("User not found")
        
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Login - Smart Water AI</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
                .card { border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="row justify-content-center align-items-center min-vh-100">
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-body p-5">
                                <div class="text-center mb-4">
                                    <i class="fas fa-tint fa-3x text-primary mb-3"></i>
                                    <h3>Sign In</h3>
                                    <div class="alert alert-danger">Invalid email or password!</div>
                                </div>
                                <form method="POST">
                                    <div class="mb-3">
                                        <label class="form-label">Email</label>
                                        <input type="email" class="form-control" name="email" required>
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">Password</label>
                                        <input type="password" class="form-control" name="password" required>
                                    </div>
                                    <div class="d-grid">
                                        <button type="submit" class="btn btn-primary">
                                            <i class="fas fa-sign-in-alt"></i> Sign In
                                        </button>
                                    </div>
                                </form>
                                <div class="text-center mt-3">
                                    <a href="/register">Don't have an account? Sign up</a>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        '''
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Login - Smart Water AI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .card { border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row justify-content-center align-items-center min-vh-100">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body p-5">
                            <div class="text-center mb-4">
                                <i class="fas fa-tint fa-3x text-primary mb-3"></i>
                                <h3>Sign In</h3>
                            </div>
                            <form method="POST">
                                <div class="mb-3">
                                    <label class="form-label">Email</label>
                                    <input type="email" class="form-control" name="email" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Password</label>
                                    <input type="password" class="form-control" name="password" required>
                                </div>
                                <div class="d-grid">
                                    <button type="submit" class="btn btn-primary">
                                        <i class="fas fa-sign-in-alt"></i> Sign In
                                    </button>
                                </div>
                            </form>
                            <div class="text-center mt-3">
                                <a href="/register">Don't have an account? Sign up</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/logout')
@login_required
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    
    greeting = get_greeting()
    usage_data = get_water_usage_data(user['id'], 14)  # 14 days for better ML
    hourly_data = get_hourly_usage_data(user['id'])
    hourly_usage, daily_usage, weekly_usage = get_current_usage_stats(user['id'])
    
    # Generate ML predictions
    ml_results, ml_error = generate_ml_predictions(user['id'])
    
    # Detect anomalies
    anomalies = detect_usage_anomalies(user['id'])
    
    # Get recent alerts
    conn = sqlite3.connect('water_monitoring.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT alert_type, message, timestamp FROM alerts 
        WHERE user_id = ? AND is_read = 0 
        ORDER BY timestamp DESC LIMIT 5
    ''', (user['id'],))
    alerts = cursor.fetchall()
    conn.close()
    
    # Convert data to JSON strings for JavaScript
    usage_data_json = json.dumps(usage_data)
    hourly_data_json = json.dumps(hourly_data)
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard - Smart Water AI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
            .card {{ border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }}
            .stat-card {{ transition: transform 0.3s ease; }}
            .stat-card:hover {{ transform: scale(1.05); }}
            .alert-notification {{
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 9999;
                min-width: 300px;
                display: none;
            }}
            .ml-card {{
                border-left: 4px solid #28a745;
                background: linear-gradient(45deg, #f8fff8, #ffffff);
            }}
            .anomaly-card {{
                border-left: 4px solid #dc3545;
                background: linear-gradient(45deg, #fff8f8, #ffffff);
            }}
            .prediction-chart {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <!-- Alert notification area -->
        <div id="alertNotification" class="alert-notification"></div>

        <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
            <div class="container">
                <a class="navbar-brand" href="/dashboard">
                    <i class="fas fa-tint text-info"></i> Smart Water AI
                </a>
                <div class="navbar-nav ms-auto">
                    <span class="navbar-text me-3">
                        <i class="fas fa-brain"></i> ML Enabled
                    </span>
                    <span class="navbar-text me-3">
                        <i class="fas fa-calendar"></i> Week: {weekly_usage:.1f}L
                    </span>
                    <a class="nav-link" href="/logout">
                        <i class="fas fa-sign-out-alt"></i> Logout ({user['name']})
                    </a>
                </div>
            </div>
        </nav>

        <div class="container mt-4">
            <!-- Greeting -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card bg-primary text-white">
                        <div class="card-body text-center">
                            <h2><i class="fas fa-brain"></i> {greeting}, {user['name']}!</h2>
                            <p class="mb-0">Advanced water monitoring with ML predictions and anomaly detection</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Stats -->
            <div class="row mb-4">
                <div class="col-md-3 mb-3">
                    <div class="card stat-card bg-info text-white">
                        <div class="card-body">
                            <h6><i class="fas fa-calendar-day"></i> Today's Usage</h6>
                            <h3 id="dailyUsageDisplay">{daily_usage:.1f}L</h3>
                            <small>{(daily_usage/user['daily_limit']*100):.1f}% of limit</small>
                            <div class="progress mt-2" style="height: 4px;">
                                <div class="progress-bar bg-light" style="width: {min(100, daily_usage/user['daily_limit']*100):.1f}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-3">
                    <div class="card stat-card bg-warning text-white">
                        <div class="card-body">
                            <h6><i class="fas fa-clock"></i> Hourly Usage</h6>
                            <h3 id="hourlyUsageDisplay">{hourly_usage:.1f}L</h3>
                            <small>{(hourly_usage/user['hourly_limit']*100):.1f}% of limit</small>
                            <div class="progress mt-2" style="height: 4px;">
                                <div class="progress-bar bg-light" style="width: {min(100, hourly_usage/user['hourly_limit']*100):.1f}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-3">
                    <div class="card stat-card bg-success text-white">
                        <div class="card-body">
                            <h6><i class="fas fa-brain"></i> ML Accuracy</h6>
                            <h3>{ml_results['model_accuracy'] if ml_results else 'N/A'}{'%' if ml_results else ''}</h3>
                            <small>Prediction model</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-3">
                    <div class="card stat-card bg-danger text-white">
                        <div class="card-body">
                            <h6><i class="fas fa-exclamation-triangle"></i> Anomalies</h6>
                            <h3>{len(anomalies)}</h3>
                            <small>Detected patterns</small>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ML Predictions Section -->
            {'<div class="row mb-4"><div class="col-12"><div class="card ml-card"><div class="card-header"><h5><i class="fas fa-brain"></i> Machine Learning Predictions - Next 5 Days</h5></div><div class="card-body">' if ml_results else ''}
            {f'''
                <div class="row">
                    <div class="col-lg-8">
                        <img src="{ml_results['chart_path']}" class="prediction-chart" alt="ML Prediction Chart">
                    </div>
                    <div class="col-lg-4">
                        <h6><i class="fas fa-chart-line"></i> Prediction Summary</h6>
                        <ul class="list-unstyled">
                            <li><strong>Model Accuracy:</strong> {ml_results['model_accuracy']}%</li>
                            <li><strong>Mean Error:</strong> ±{ml_results['mean_error']}L</li>
                            <li><strong>Trend:</strong> {ml_results['trend'].title()}</li>
                            <li><strong>Daily Change:</strong> {ml_results['daily_change']:+.1f}L/day</li>
                        </ul>
                        
                        <h6 class="mt-3"><i class="fas fa-calendar-alt"></i> Next 5 Days</h6>
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr><th>Date</th><th>Predicted</th></tr>
                                </thead>
                                <tbody>
                                    {"".join([f'<tr><td>{datetime.strptime(p["date"], "%Y-%m-%d").strftime("%m/%d")}</td><td>{p["predicted_usage"]}L</td></tr>' for p in ml_results['predictions']])}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            ''' if ml_results else f'<div class="alert alert-warning"><i class="fas fa-exclamation-triangle"></i> {ml_error}</div>'}
            {'</div></div></div></div>' if ml_results else ''}

            <!-- Anomaly Detection Section -->
            {f'''
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card anomaly-card">
                        <div class="card-header">
                            <h5><i class="fas fa-exclamation-triangle"></i> Anomaly Detection - Unusual Patterns</h5>
                        </div>
                        <div class="card-body">
                            {"".join([f'''
                            <div class="alert alert-{'danger' if anomaly['severity'] == 'high' else 'warning'} mb-2">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <strong>{anomaly['type']}</strong> - {anomaly['date']}
                                        {'<small> at ' + anomaly['hour'] + ':00</small>' if 'hour' in anomaly else ''}
                                        <br><small>{anomaly['message']}</small>
                                    </div>
                                    <div class="text-end">
                                        <strong>{anomaly['usage']:.1f}L</strong>
                                    </div>
                                </div>
                            </div>
                            ''' for anomaly in anomalies]) if anomalies else '<div class="text-center text-success"><i class="fas fa-check-circle fa-3x mb-3"></i><p>No anomalies detected</p><small>Your usage patterns are normal!</small></div>'}
                        </div>
                    </div>
                </div>
            </div>
            ''' if anomalies or True else ''}

            <!-- Charts Row -->
            <div class="row mb-4">
                <div class="col-lg-8">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="fas fa-chart-line"></i> 14-Day Water Usage History</h5>
                        </div>
                        <div class="card-body">
                            <canvas id="usageChart" height="100"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-lg-4">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="fas fa-clock"></i> Today's Hourly Pattern</h5>
                        </div>
                        <div class="card-body">
                            <canvas id="hourlyChart" height="150"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Alerts Row -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="fas fa-bell"></i> Recent System Alerts</h5>
                        </div>
                        <div class="card-body" style="max-height: 200px; overflow-y: auto;">
                            {"".join([f'<div class="alert alert-warning alert-sm mb-2"><small><strong>Alert:</strong><br>{alert[1]}</small></div>' for alert in alerts]) if alerts else '<div class="text-center text-muted"><i class="fas fa-check-circle fa-3x mb-3"></i><p>No recent alerts</p><small>Your water usage is within limits!</small></div>'}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Add Usage -->
            <div class="row">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="fas fa-plus-circle"></i> Add Water Usage</h5>
                        </div>
                        <div class="card-body">
                            <form id="usageForm" class="row g-3">
                                <div class="col-md-6">
                                    <label class="form-label">Usage Amount (Liters)</label>
                                    <input type="number" class="form-control" id="usageAmount" step="0.1" min="0" required>
                                </div>
                                <div class="col-md-6 d-flex align-items-end">
                                    <button type="submit" class="btn btn-primary me-2" id="addUsageBtn">
                                        <i class="fas fa-plus"></i> Add Usage
                                    </button>
                                    <button type="button" class="btn btn-outline-info me-2" onclick="simulateUsage(5)">
                                        <i class="fas fa-flask"></i> Test 5L
                                    </button>
                                    <button type="button" class="btn btn-outline-warning me-2" onclick="simulateUsage(15)">
                                        <i class="fas fa-exclamation-triangle"></i> Test 15L
                                    </button>
                                    <button type="button" class="btn btn-outline-danger" onclick="simulateUsage(25)">
                                        <i class="fas fa-fire"></i> Test 25L
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Historical Usage Chart (14 days)
            const ctx = document.getElementById('usageChart').getContext('2d');
            const usageData = {usage_data_json};
            
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: usageData.map(item => item[0]),
                    datasets: [{{
                        label: 'Daily Usage (L)',
                        data: usageData.map(item => item[1]),
                        borderColor: '#0d6efd',
                        backgroundColor: 'rgba(13, 110, 253, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointBackgroundColor: '#0d6efd',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }}, {{
                        label: 'Daily Limit',
                        data: Array(usageData.length).fill({user['daily_limit']}),
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'top',
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false,
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Usage (Liters)'
                            }}
                        }}
                    }}
                }}
            }});

            // Hourly Usage Chart
            const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
            const hourlyData = {hourly_data_json};
            const hourLabels = Array.from({{length: 24}}, (_, i) => `${{i}}:00`);
            
            new Chart(hourlyCtx, {{
                type: 'bar',
                data: {{
                    labels: hourLabels,
                    datasets: [{{
                        label: 'Hourly Usage (L)',
                        data: hourlyData,
                        backgroundColor: 'rgba(255, 193, 7, 0.8)',
                        borderColor: '#ffc107',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Liters'
                            }}
                        }}
                    }}
                }}
            }});

            // Show notification function
            function showNotification(message, type = 'info') {{
                const alertDiv = document.getElementById('alertNotification');
                alertDiv.className = `alert alert-${{type}} alert-dismissible fade show alert-notification`;
                alertDiv.innerHTML = `
                    ${{message}}
                    <button type="button" class="btn-close" onclick="hideNotification()"></button>
                `;
                alertDiv.style.display = 'block';
                
                setTimeout(() => {{
                    hideNotification();
                }}, 5000);
            }}

            function hideNotification() {{
                const alertDiv = document.getElementById('alertNotification');
                alertDiv.style.display = 'none';
            }}

            // Form submission
            document.getElementById('usageForm').addEventListener('submit', async function(e) {{
                e.preventDefault();
                
                const submitButton = document.getElementById('addUsageBtn');
                const originalText = submitButton.innerHTML;
                const amount = document.getElementById('usageAmount').value;
                
                if (!amount || parseFloat(amount) <= 0) {{
                    showNotification('Please enter a valid usage amount!', 'warning');
                    return;
                }}
                
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Adding...';
                submitButton.disabled = true;
                
                try {{
                    const response = await fetch('/api/usage', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ amount: parseFloat(amount) }})
                    }});
                    
                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}
                    
                    const result = await response.json();
                    
                    if (result.success) {{
                        showNotification(`Successfully added ${{amount}}L! ML predictions will be updated.`, 'success');
                        
                        document.getElementById('usageAmount').value = '';
                        
                        if (result.alerts && result.alerts.length > 0) {{
                            result.alerts.forEach(alert => {{
                                const alertType = alert.severity === 'high' ? 'danger' : 'warning';
                                setTimeout(() => {{
                                    showNotification(alert.message, alertType);
                                }}, 1000);
                            }});
                        }}
                        
                        setTimeout(() => {{
                            location.reload();
                        }}, 3000);
                    }} else {{
                        showNotification(result.error || 'Failed to add usage', 'danger');
                    }}
                }} catch (error) {{
                    console.error('Error:', error);
                    showNotification(`Error adding usage: ${{error.message}}`, 'danger');
                }} finally {{
                    submitButton.innerHTML = originalText;
                    submitButton.disabled = false;
                }}
            }});

            function simulateUsage(amount) {{
                document.getElementById('usageAmount').value = amount;
                showNotification(`Set test usage to ${{amount}}L. Click "Add Usage" to submit.`, 'info');
            }}
        </script>
    </body>
    </html>
    '''

@app.route('/api/usage', methods=['POST'])
@login_required
def add_usage():
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data or 'amount' not in data:
            return jsonify({'success': False, 'error': 'Missing amount in request'}), 400
        
        usage_amount = float(data['amount'])
        if usage_amount <= 0:
            return jsonify({'success': False, 'error': 'Usage amount must be positive'}), 400
        
        user = get_current_user()
        if not user:
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
        
        print(f"Adding {usage_amount}L usage for user {user['name']} (ID: {user['id']})")
        
        # Add usage to database
        conn = sqlite3.connect('water_monitoring.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO water_usage (user_id, usage_amount)
            VALUES (?, ?)
        ''', (user['id'], usage_amount))
        conn.commit()
        conn.close()
        
        print(f"Successfully added {usage_amount}L usage to database")
        
        # Check for alerts
        alerts = check_usage_alerts(user['id'], usage_amount)
        print(f"Generated {len(alerts)} alerts")
        
        return jsonify({
            'success': True, 
            'alerts': alerts,
            'message': f'Added {usage_amount}L successfully'
        })
        
    except ValueError as e:
        print(f"ValueError in add_usage: {e}")
        return jsonify({'success': False, 'error': 'Invalid usage amount format'}), 400
    except Exception as e:
        print(f"Error in add_usage: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    print("🌊 Smart Water AI Monitoring System Starting...")
    print("📍 Access the app at: http://127.0.0.1:5000")
    print("🧠 ML Features: Predictions, Anomaly Detection, Advanced Analytics")
    print("📊 Features: 14-day history, Trend analysis, Static chart generation")
    print("🎯 Enhanced: scikit-learn ML, matplotlib charts, pandas analytics")
    app.run(debug=True, host='127.0.0.1', port=5000)
