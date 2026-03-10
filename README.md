# CNN-BASED-CHEST-RADIOGRAPH-CLASSIFIER-FOR-COVID-19-IDENTIFICATION




**Department:** Artificial Intelligence and Data Science  
**Batch:** 2022 – 2026  

---

## Abstract
This project presents an AI-powered system to detect COVID-19 from chest X-ray images using Convolutional Neural Networks (CNNs). Pre-trained models (ResNet50, VGG16, Xception) are fine-tuned using transfer learning for high accuracy, even on limited datasets. The models are integrated into a Django-based web application, allowing users to upload X-rays and receive real-time diagnostic predictions, reducing workload and improving consistency for healthcare professionals.

---

## Features
- Automated COVID-19 detection using CNNs  
- Transfer learning for improved accuracy  
- Ensemble of ResNet50, VGG16, and Xception models  
- Django web interface for image upload and real-time results  
- Confidence score and user-friendly outputs  

---

## System Architecture
1. **Frontend:** Upload chest X-ray images via web interface.  
2. **Preprocessing:** Resize images to 224×224, normalize, augment during training.  
3. **Model Inference:** Predict COVID or Non-COVID using CNN models.  
4. **Backend:** Django handles model loading, prediction, and response.  
5. **Result Display:** Shows predicted class and confidence score.  

---

## Dataset
- **Source:** SARS-COV-2 CT-Scan Dataset (Kaggle)  
- **Composition:** COVID (positive) and Non-COVID (negative) cases  
- **Preprocessing:** Resizing, normalization, noise removal, augmentation  
- **Splitting:** 70% training, 15% validation, 15% testing  

---

## Results
| Model      | Accuracy |
|------------|----------|
| ResNet50   | 95%      |
| VGG16      | 93%      |
| Xception   | 94%      |
**Ensemble:** 96%  

---

## Advantages
- Fast, automated COVID-19 detection  
- Reduces diagnostic workload  
- High accuracy and reliability  
- Scalable and web-deployable  
- Can be extended to other lung diseases  

---

## Requirements
- Python 3.10  
- Django  
- TensorFlow & Keras  
- VS Code  
- Libraries: OpenCV, Pillow, NumPy, pandas  

---

## Future Enhancements
- Increase dataset diversity for better generalization  
- Grad-CAM visualization for interpretability  
- Multi-disease detection  
- Multi-language web support  

---

## Conclusion
The system demonstrates the practical application of CNNs for COVID-19 detection with real-time web deployment. It reduces manual effort, improves diagnostic accuracy, and serves as a scalable healthcare tool.


----
##DIRECTORY STRUCTURE
COVID_CNN_Classifier/
│
├── manage.py
├── README.md
├── requirements.txt
├── db.sqlite3
├── static/
│   └── models/        # Trained CNN models (.hdf5, .json)
├── media/             # Uploaded X-ray images
├── templates/
│   └── app/           # HTML templates
├── app/               # Django app
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   └── ...
└── dataset/           # Original/preprocessed dataset (into COVID/non-COVID folders under dataset folder)-->CHEST XRAY IMG DATASET FROM KAGGLE



