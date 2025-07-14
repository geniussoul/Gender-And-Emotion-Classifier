👤 Gender, Age & Emotion Classifier 🎭
======================================

A Python-based deep learning application that performs real-time face detection and predicts a person’s 
gender, age group, and emotion from either a webcam feed or a selected image.

This project uses pre-trained models via OpenCV's DNN module, Haar Cascades, and ONNX for interactive facial analysis.

--------------------------
🚀 Features
--------------------------
✅ Real-time face detection using webcam  
✅ Static image input via file dialog  
✅ Gender classification (Male / Female)  
✅ Age group prediction (e.g., 15–20, 25–32, etc.)  
✅ Emotion recognition (Happy, Sad, Surprise, Neutral, etc.)  
✅ Clean and intuitive OpenCV UI

--------------------------
🧠 Models Used
--------------------------
Task             | Model Type  | Format                    | Framework
-----------------|-------------|---------------------------|----------
Face Detection   | Haar Cascade| .xml                      | OpenCV
Gender Prediction| CNN         | .prototxt + .caffemodel   | Caffe
Age Estimation   | CNN         | .prototxt + .caffemodel   | Caffe
Emotion Detection| ResNet50    | .onnx                     | ONNX

--------------------------
📁 Project Structure
--------------------------
📦 GenderAgeEmotionClassifier
│
├── gender_age_emotion_classifier.py
├── models/
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│   └── FER_static_ResNet50_AffectNet.onnx

--------------------------
📦 Installation
--------------------------
1️⃣ Clone the Repository:
    git clone https://github.com/your-username/GenderAgeEmotionClassifier.git
    cd GenderAgeEmotionClassifier

2️⃣ Install Required Libraries:
    pip install opencv-python numpy

Note: Tkinter is usually pre-installed with Python.

--------------------------
⚙️ How to Run
--------------------------
📽️ Option 1: Webcam Input (Live Detection)
    python gender_age_emotion_classifier.py
    ➤ Enter 1 when prompted
    ➤ A window opens with live face tracking
    ➤ Press 'q' to quit

🖼️ Option 2: Image Input
    python gender_age_emotion_classifier.py
    ➤ Enter 2 when prompted
    ➤ Select an image file
    ➤ Output will show labeled faces

--------------------------
🔖 Label Mappings
--------------------------
Gender    | Age Group | Emotion
----------|-----------|---------
Male      | (0–2)     | Neutral
Female    | (4–6)     | Sad
          | (8–12)    | Happy
          | (15–20)   | Surprise
          | (25–32)   | Fear
          | (38–43)   | Disgust
          | (48–53)   | Contempt
          | (60–100)  | 

--------------------------
🔧 Notes
--------------------------
- All model paths are hardcoded in the script. Adjust them based on your system.
- Emotion model uses 224x224 input; age/gender use 227x227.
- Performance can improve by enabling OpenCV's CUDA backend if available.

--------------------------
📜 License
--------------------------
This project is licensed under the MIT License — free to use, modify, and distribute.

--------------------------
🙌 Acknowledgements
--------------------------
- OpenCV
- Caffe Models (by spmallick)
- AffectNet & FER+ Dataset
- Built with ❤️ by Martand Mishra

--------------------------
📬 Contact
--------------------------
📧 martandmishra473@gmail.com  
🌐 LinkedIn: https://linkedin.com/in/your-link

--------------------------
✨ Quote
--------------------------
"Machines see faces, but with code — they read stories."
