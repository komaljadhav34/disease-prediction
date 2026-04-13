# =====================
# Step-by-Step Flask App for Leaf Disease Detection (NumPy CNN)
# =====================

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from PIL import Image
import numpy as np
import os
import re

# -------------------------------------
# Step 1: Initialize Flask App
# -------------------------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generated secure random key for this session

# -------------------------------------
# Step 2: In-Memory User Storage (Replace with database in production)
# -------------------------------------
users = {}  # Dictionary to store username:password pairs

# -------------------------------------
# Step 2: Define CNN Feature Extractors (NumPy)
# -------------------------------------
kernels = [
    np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
    np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
    np.array([[0, -1, 1], [-1, 0, 1], [1, 1, 0]])
]

def conv2d(image, kernel):
    kh, kw = kernel.shape
    ih, iw = image.shape
    out = np.zeros((ih - kh + 1, iw - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return out

def relu(x):
    return np.maximum(0, x)

def max_pooling(image, size=2):
    h, w = image.shape
    pooled = image[:h - h % size, :w - w % size].reshape(h // size, size, w // size, size)
    return pooled.max(axis=(1, 3))

def color_histogram(img, bins=8):
    hists = [np.histogram(img[..., c], bins=bins, range=(0, 1))[0] for c in range(3)]
    return np.concatenate(hists)

def extract_features(img, kernels):
    feats = []
    for c in range(3):
        ch = img[:, :, c]
        for k in kernels:
            conv = relu(conv2d(ch, k))
            pool = max_pooling(conv)
            feats.append(pool.flatten())
    return np.concatenate(feats + [color_histogram(img)])

# -------------------------------------
# Step 3: Load Dataset Once on Startup
# -------------------------------------
dataset_path = 'dataset'  # <- this folder must exist with subfolders for each class
image_size = (64, 64)
dataset_features = []

def label_from_folder(folder):
    return folder.replace("Tomato___", "").replace("_", " ").title()

for folder in os.listdir(dataset_path):
    full_folder = os.path.join(dataset_path, folder)
    if not os.path.isdir(full_folder): continue
    label = label_from_folder(folder)

    for fname in os.listdir(full_folder):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            fpath = os.path.join(full_folder, fname)
            img = Image.open(fpath).resize(image_size)
            img = np.array(img).astype(np.float32) / 255.0
            feat = extract_features(img, kernels)
            dataset_features.append((feat, label))
    print(f"Loaded: {label}")

# -------------------------------------
# Step 4: Define Remedies Dictionary
# -------------------------------------
remedies = {
    "Bacterial Spot": "Use copper-based bactericides.",
    "Early Blight": "Remove infected leaves and use fungicides.",
    "Late Blight": "Apply appropriate fungicide treatment.",
    "Leaf Mold": "Ensure good air circulation and use fungicides.",
    "Septoria Leaf Spot": "Prune affected areas and apply fungicide.",
    "Healthy": "No remedy needed. The plant is healthy."
}

# -------------------------------------
# Step 5: Password Validation Function
# -------------------------------------
def is_valid_password(password):
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True

# ... (previous imports and code remain the same until routes)

# -------------------------------------
# Step 5: Define Routes
# -------------------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if name in users:
            flash("Username already exists!")
            return redirect(url_for('signup'))

        if not is_valid_password(password):
            flash("Password must be at least 8 characters long and contain at least one capital letter, one number, and one special symbol (!@#$%^&*(),.?\":{}|<>)!")
            return redirect(url_for('signup'))

        users[name] = password  # Store the username and password (hash in production)
        flash(f"Account created successfully for {name}!")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']

        if name not in users or users[name] != password:
            flash("Invalid username or password!")
            return redirect(url_for('login'))

        if not is_valid_password(password):
            flash("Password does not meet security requirements!")
            return redirect(url_for('login'))

        flash("Login successful!")
        return redirect(url_for('detect'))  

    return render_template('login.html')

@app.route('/logout')
def logout():
    flash("You have been logged out.")
    return redirect(url_for('home'))  

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        img = Image.open(file.stream).resize(image_size)
        img = np.array(img).astype(np.float32) / 255.0
        input_feat = extract_features(img, kernels)

        distances = [(np.linalg.norm(input_feat - feat), label) for feat, label in dataset_features]
        best_match = min(distances, key=lambda x: x[0])
        predicted_label = best_match[1]

        return jsonify({
            'result': predicted_label,
            'remedy': remedies.get(predicted_label, "No remedy found.")
        })

# Disease-specific routes
@app.route('/leaf-blight')
def leaf_blight():
    return render_template('leaf-blight.html', disease_name="Leaf Blight", remedy=remedies.get("Leaf Blight"))

@app.route('/downymildew')
def downymildew():
    return render_template('downymildew.html', disease_name="Downy Mildew", remedy=remedies.get("Downy Mildew"))

@app.route('/mosaicvirus')
def mosaicvirus():
    return render_template('mosaicvirus.html', disease_name="Mosaic Virus", remedy=remedies.get("Mosaic Virus"))

@app.route('/powderymildew')
def powderymildew():
    return render_template('powderymildew.html', disease_name="Powdery Mildew", remedy=remedies.get("Powdery Mildew"))

@app.route('/rust')
def rust():
    return render_template('rust.html', disease_name="Rust", remedy=remedies.get("Rust"))

@app.route('/canker')
def canker():
    return render_template('canker.html', disease_name="Canker", remedy=remedies.get("Canker"))

@app.route('/clubroot')
def clubroot():
    return render_template('clubroot.html', disease_name="Clubroot", remedy=remedies.get("Clubroot"))

@app.route('/anthracnose')
def anthracnose():
    return render_template('anthracnose.html', disease_name="Anthracnose", remedy=remedies.get("Anthracnose"))

@app.route('/sootymold')
def sootymold():
    return render_template('sootymold.html', disease_name="Sooty Mold", remedy=remedies.get("Sooty Mold"))

@app.route('/wilt')
def wilt():
    return render_template('wilt.html', disease_name="Wilt", remedy=remedies.get("Wilt"))

# ... (rest of the code remains the same)

# -------------------------------------
# Step 6: Run the App
# -------------------------------------
if __name__ == '__main__':
    app.run(debug=True)