# Metalic-surface-defect-detection

Purpose of the project is to detect the defects on a metalic surface. The dataset was taken from Kaggle https://www.kaggle.com/datasets/alex000kim/gc10det 

Dataset is stored in the dataset folder divided in train, test and valid folders with images and labels. 

There is a requirements.txt file in the directory run that to install all the neccesary libraries to run the project. 

commond to install all the libraries : pip install -r requirements.txt

Steps to run the project :
1. Run the defect_detection to train the model
2. Test the performance of the model by running the model_prediction and predict_multiple functions
3. Run the video_testing.py to see how it works on video

model folder containes best.pt that is the best model to predict values.