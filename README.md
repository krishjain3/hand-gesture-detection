Hand Gesture Detection System  
Real-Time Computer Vision and Machine Learning Application  

The Hand Gesture Detection System is a real-time application designed to recognize and classify hand gestures using a webcam. It uses computer vision to extract hand features and machine learning to predict gestures, enabling a simple and touchless way of interaction.

---

Project Summary  

The application enables users to:

• Capture hand gesture data using a webcam  
• Create a custom dataset of hand landmarks  
• Train a machine learning model for gesture classification  
• Perform real-time gesture detection  
• Display predictions along with confidence scores  

The system uses MediaPipe for hand landmark detection and a Gradient Boosting model for accurate gesture classification.

---

Technology Stack  

Computer Vision: OpenCV, MediaPipe  
Machine Learning: Scikit-learn (Gradient Boosting)  
Programming Language: Python  
Data Processing: NumPy, Pandas  
Visualization: Matplotlib  
Model Storage: Joblib  
Version Control: Git, GitHub  

---

Key Features  

Real-Time Gesture Detection  
• Detects hand gestures using webcam input  
• Displays predictions instantly on screen  

Custom Dataset Collection  
• Captures 21 hand landmark coordinates (x, y, z)  
• Supports multiple gesture classes  

Machine Learning Classification  
• Uses Gradient Boosting for improved accuracy  
• Handles similar gesture variations better  

Confidence-Based Prediction  
• Shows prediction along with confidence percentage  
• Filters uncertain predictions  

Improved Data Processing  
• Normalizes landmark positions  
• Improves real-time stability and accuracy  

---

Project Workflow  

Data Collection  
• Capture gesture data using webcam  
• Store landmark coordinates in CSV format  

Model Training  
• Train machine learning model using collected data  
• Evaluate accuracy and performance  

Real-Time Detection  
• Use trained model to predict gestures  
• Display results live through webcam feed  

---

Project Structure (High-Level)  

hand-gesture-detection-system/  

data/        → Dataset (not uploaded)  
models/      → Trained model files (not uploaded)  
src/         → Source code  
  collect_data.py  
  train_model.py  
  detect.py  

README.md  
requirements.txt  
.gitignore  

---

How to Run  

• Clone the repository  
  git clone https://github.com/krishjain3/hand-gesture-detection.git  

• Install dependencies  
  pip install -r requirements.txt  

• Run the application  
  python src/detect.py  

---

Challenges Faced  

• Confusion between similar gestures (e.g., peace and pointing)  
• Variations in lighting and hand position  
• Low confidence in initial real-time predictions  

---

Improvements Made  

• Increased dataset size for better training  
• Collected data with more variation (angle, distance, lighting)  
• Normalized hand landmark positions  
• Switched from Random Forest to Gradient Boosting  
• Added confidence-based filtering  

---

Conclusion  

This project demonstrates how computer vision and machine learning can be combined to create an interactive real-time system. It highlights the importance of data quality, feature engineering, and model selection in achieving reliable performance.

---

Author  

Krish Jain  
B.Tech Computer Science Engineering (AI & ML)  
VIT Bhopal  