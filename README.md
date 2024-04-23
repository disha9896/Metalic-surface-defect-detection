# Metalic-surface-defect-detection

The project aims to detect the defects on a metallic surface. The dataset was taken from Kaggle https://www.kaggle.com/datasets/alex000kim/gc10det 

The dataset is stored in the dataset folder divided into train, test and valid folders with images and labels. 

## Setup Instructions
1. Make sure you have python installed in your system
2. Install required libraries by running following code. requirements.txt file in present in the repository
   ```
    pip install -r requirements.txt 
    ```
3. Install git lfs to use the large files i.e. models in the repository.
Homebrew:  ```
            brew install git-lfs
           ```
           ```
MacPorts:    port install git-lfs
            ```
Then pull from the git repo using the command :
```
git lfs pull
```



Steps to run the project :
1. Run the defect_detection to train the model
2. Test the performance of the model by running the model_prediction and predict_multiple functions
3. Run the video_testing.py to see how it works on video

model folder containes best.pt that is the best model to do prediction. Yolov9e.pt is the pre trained model 

Steps to train the model :
1. Add the dataset directory path.
2. Load the pre-trained yolov9e.pt model from the model folder.
3. Run the train function to train the model. 
4. It will store the validation resuls, detected results and the model in the run/detect/train folder.

Use the best.pt model to predict results as it is trained properly with all the images the dataset here is just a sample dataset for you to try and train the model whole dataset link is provided above. The training for complete dataset requires GPU and takes more time to train.
