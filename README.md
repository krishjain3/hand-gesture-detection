Hand Gesture Detection System  
Real-Time Computer Vision and Machine Learning Application  

The Hand Gesture Detection System is a real-time application designed to recognize and classify hand gestures using a webcam. It combines computer vision and machine learning techniques to enable touchless interaction with a system. This project was developed to explore practical applications of gesture-based interfaces using landmark detection and classification models.


Project Summary  
The application allows users to:

 . Capture hand gesture data using a webcam  

 . Generate a custom dataset of hand landmarks  

 . Train a machine learning model for gesture classification  

 . Recognize gestures in real-time with confidence scores  

 . Display predictions live on screen  

 . The system uses MediaPipe for extracting hand landmarks and a Gradient Boosting model for accurate gesture classification.


Technology Stack  
Computer Vision: OpenCV, MediaPipe  
Machine Learning: Scikit-learn (Gradient Boosting)  
Programming Language: Python  
Data Processing: NumPy, Pandas  
Visualization: Matplotlib  
Model Storage: Joblib  
Version Control: Git, GitHub  


Key Features  

Real-Time Gesture Detection  
 . Detects hand gestures using webcam input and displays predictions instantly  

Custom Dataset Collection  
 . Captures 21 hand landmark coordinates (x, y, z) for each gesture  

Machine Learning Classification  
 . Uses Gradient Boosting for accurate and reliable gesture prediction  

Confidence-Based Prediction  
 . Displays prediction along with confidence percentage  

Improved Data Processing  
 . Applies normalization to hand landmarks for better real-time accuracy  


Project Workflow  

Data Collection  
 . Capture gesture data using webcam and store landmark values  

Model Training  
 . Train a machine learning model using collected dataset  

Real-Time Detection  
 . Use trained model to predict gestures from live webcam feed  


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


How to Run  

1. Clone the repository  
git clone https://github.com/krishjain3/hand-gesture-detection.git  
cd hand-gesture-detection  

2. Create virtual environment  
python -m venv venv  
venv\Scripts\activate  

3. Install dependencies  
pip install -r requirements.txt  

4. Run the application  
python src/detect.py  


Challenges Faced  

 . Confusion between similar gestures (e.g., peace and pointing)  

 . Variation in lighting and hand position affecting predictions  

 . Low confidence in initial real-time results  


Improvements Made  

 . Increased dataset size for better training  

 . Normalized landmark coordinates to improve accuracy  

 . Upgraded model from Random Forest to Gradient Boosting  

 . Added confidence-based filtering for predictions  


Conclusion  

This project demonstrates how computer vision and machine learning can be integrated to create an interactive real-time system. It highlights the importance of data quality, feature representation, and model selection in achieving reliable performance.


Author  
Krish Jain  
B.Tech Computer Science Engineering (AI & ML)  
VIT Bhopal  