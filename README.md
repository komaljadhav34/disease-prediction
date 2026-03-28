# Leaf Disease Detection System

A Flask-based web application that identifies plant leaf diseases using a custom Convolutional Neural Network (CNN) feature extraction implemented with NumPy.

## 🌟 Features

- **Disease Detection**: Upload an image of a plant leaf to identify the disease and get remedy suggestions.
- **Custom CNN implementation**: Feature extraction (convolution, ReLU, max pooling) implemented from scratch using NumPy.
- **Nearest-Neighbor Classification**: Diseases are identified by comparing image features against a pre-loaded dataset using Euclidean distance.
- **User Authentication**: Secure signup and login system with password validation (requires at least 8 characters, one uppercase letter, one number, and one special character).
- **Disease Information**: Extensive informational pages for various diseases like Leaf Blight, Downy Mildew, Anthracnose, etc.
- **Responsive Web UI**: A modern interface for easy interaction and navigation.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Data Processing**: NumPy, Pillow (PIL)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## 🚀 Getting Started

### Prerequisites

- Python 3.x installed on your system.

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd leaf-disease-detection
    ```

2.  Install the required dependencies:
    ```bash
    pip install flask numpy pillow
    ```

## 📊 Dataset

The project uses the **Tomato Leaf Disease Dataset** sourced from **Kaggle** (part of the larger **PlantVillage** dataset). It contains high-quality images of tomato leaves categorized into various health conditions and diseases.

You can find the original dataset on Kaggle: [Tomato Leaf Disease Dataset](https://www.kaggle.com/datasets/kaustubhb99/tomato-leaf-disease-dataset) (or similar PlantVillage-sourced datasets).

### Dataset Preparation

The application requires a `dataset/` folder in the root directory. This folder should contain subdirectories, each named after a disease class (e.g., `Bacterial Spot`, `Healthy`). Images within these folders are used to train the nearest-neighbor classifier on startup.

Example structure:
```text
dataset/
├── Bacterial_Spot/
│   ├── image1.jpg
│   └── image2.jpg
├── Healthy/
│   ├── image3.jpg
│   └── image4.jpg
...
```

### Running the Application

1.  Start the Flask server:
    ```bash
    python app.py
    ```

2.  Open your browser and navigate to `http://127.0.0.1:5000`.

## 📁 Project Structure

- `app.py`: Main Flask application containing backend logic, CNN feature extraction, and routing.
- `static/`: Contains static assets like CSS (`style.css`), images, and client-side JavaScript.
- `templates/`: HTML templates for different pages of the application.
- `dataset/`: Directory for storing training images organized by disease class.
- `*.html`: Individual informational pages for specific diseases located in the root (legacy or specific use).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
