import os
import io
import numpy as np
from datetime import datetime
from flask import (
    Flask, request, render_template, url_for, jsonify,
    redirect, flash
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# --- Keras 3 Imports ---
from keras.models import load_model
from keras.utils import load_img, img_to_array

# ----------------------------
# 1. FLASK APP & DB CONFIG
# ----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_that_you_should_change'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'site.db')
db = SQLAlchemy(app)

# ----------------------------
# 2. LOGIN MANAGER CONFIG
# ----------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'signin'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info' 

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ----------------------------
# 3. DATABASE MODELS (TABLES)
# ----------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)
    scans = db.relationship('Scan', backref='author', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# ----------------------------
# 4. KERAS MODEL LOADING
# ----------------------------
MODEL_PATH = 'model/best_vgg19_lung.h5' 
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    'Bengin cases', 'Malignant cases', 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 'normal', 
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
]

print(f"Loading model from {MODEL_PATH}...")
try:
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. Error: {e}")
    model = None

# --- Simplified Preprocessing ---
def preprocess_image(img_bytes):
    """
    Pre-processes image bytes for VGG19 (1/255.0 rescale).
    """
    img = load_img(io.BytesIO(img_bytes), target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    # Preprocessing for VGG19 (1/255.0 rescale)
    processed_img = img_array_expanded / 255.0 
    
    return processed_img

# ----------------------------
# 5. AUTHENTICATION ROUTES
# ----------------------------
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password.', 'danger')
    return render_template('signin.html', title='Sign In')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists.', 'warning')
            return redirect(url_for('signup'))
        
        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        if new_user.id == 1:
            new_user.is_admin = True
            db.session.commit()
            flash('Your ADMIN account has been created! Please sign in.', 'success')
        else:
            flash('Your account has been created! Please sign in.', 'success')
        
        return redirect(url_for('signin'))
    return render_template('signup.html', title='Sign Up')

@app.route('/signout')
@login_required
def signout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('landing'))

# ----------------------------
# 6. APPLICATION ROUTES
# ----------------------------
@app.route('/')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('landing.html', title='Welcome')

@app.route('/home')
@login_required
def home():
    return render_template('home.html', title='Home')

@app.route('/analyzer')
@login_required
def analyzer():
    return render_template('analyzer.html', title='Analyzer')

@app.route('/details/<string:class_name>')
@login_required
def details(class_name):
    return render_template('details.html', title=class_name.title(), class_name=class_name)

@app.route('/history')
@login_required
def history():
    scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).all()
    return render_template('history.html', title='Scan History', scans=scans)

# --- SIMPLIFIED: Admin Route ---
@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('home'))
        
    # Get all users to list (no stats)
    users = User.query.all()
    
    return render_template(
        'admin.html', 
        title='Admin Dashboard', 
        users=users
    )

# ----------------------------
# 7. API ROUTE (for JavaScript)
# ----------------------------
# --- SIMPLIFIED: Predict API ---
@app.route('/predict_api', methods=['POST'])
@login_required
def predict_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files['file']
    if f.filename == '' or model is None:
        return jsonify({"error": "No selected file or model not loaded"}), 500

    try:
        img_bytes = f.read()
        
        # 1. Preprocess
        processed_img = preprocess_image(img_bytes)
        
        # 2. Make prediction
        predictions = model.predict(processed_img)
        predicted_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_index]
        confidence = float(predictions[0][predicted_index] * 100)
        
        # 3. Save to database
        new_scan = Scan(
            filename=f.filename,
            prediction=predicted_class_name,
            confidence=confidence,
            author=current_user
        )
        db.session.add(new_scan)
        db.session.commit()
        
        # 4. --- Return SIMPLE JSON Response ---
        return jsonify({
            "prediction": predicted_class_name,
            "confidence": round(confidence, 2)
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# ----------------------------
# 8. RUN THE APP
# ----------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)