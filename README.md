# Face Recognition using CNN

This project implements a **Face Recognition system using Convolutional Neural Networks (CNN)**.  
It is built using Python and deep learning techniques to detect and recognize human faces from images.

# View App: [Hugging Face Space](https://huggingface.co/spaces/venmugilrajan/FACE_RECOGNIZATION_CNN)
---

## 🚀 Features
- Face detection and recognition
- CNN-based deep learning model
- Training and testing pipeline
- Modular project structure
- Easy to run and extend

---

## 📁 Project Structure
```


face_recognition_cnn/
│── app.py # Main application file
│── requirements.txt # Python dependencies
│── train/ # Training scripts and dataset
│── test/ # Testing scripts and data
│── .gitignore # Ignored files

````

---

## 🛠️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/venmugilrajan/face_recognition_cnn.git
cd face_recognition_cnn
````

### 2️⃣ Create and activate virtual environment

```bash
conda create -n tenv python=3.10
conda activate tenv
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the application using:

```bash
python app.py
```

Make sure required datasets and model files are properly placed before execution.

---

## 🧠 Model Details

* Architecture: Convolutional Neural Network (CNN)
* Framework: PyTorch / TensorFlow (based on implementation)
* Trained model files are **not included** in this repository due to size limitations.

> 📌 Model files (`.pth`) should be stored externally (Google Drive / Git LFS).

---

## 📊 Dataset

* Dataset should be organized inside the `train/` and `test/` directories.
* Ensure correct labeling before training.

---

## 📌 Requirements

* Python 3.8+
* NumPy
* OpenCV
* PyTorch / TensorFlow
* Other dependencies listed in `requirements.txt`

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Venmugil Rajan S**
- GitHub: [@venmugilrajan](https://github.com/venmugilrajan)

---
