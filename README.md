# Metalic-surface-defect-detection

The goal of this project is to develop an automated inspection process with precise localization of the defects on the metallic surface for example coffee machine, microwave and other small appliances. The defects that can appear are of many types such as dents, scratches or stains, burns, rust or cracks on the metallic surface. Currently we will be focusing on coffee machine to detect defects. 


## Dataset

Training of the defect detection was done using the kaggle dataset which can be downloaded using this link https://www.kaggle.com/datasets/alex000kim/gc10det. Scratch detection dataset was downloaded from https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data/data this dataset did not have labels so need to annotate them. 

A sample dataset from the original dataset is stored in the dataset folder which is divided into train, test and valid folder. It also contains the labels(annotations) and data.yml file which help to locate the images and the labels for training.

This sample dataset will help to train a sample model.

## Setup Instructions
1. Make sure you have python installed in your system
2. Install required libraries by running following code. requirements.txt file present in the repository
   ```
    pip install -r requirements.txt 
    ```
3. Install git lfs to use the large files i.e. models in the repository.
   Homebrew:  `brew install git-lfs`
   MacPorts: `port install git-lfs`
            
   Then pull from the git repo using the command :
   ```
   git lfs pull
   ```

## File Information
1. Run the train_demo to train the model
2. Test the performance of the model by running the model_prediction and predict_multiple functions
3. Run the video_testing.py to see how it works on video
4. The model folder contains the best.pt which is the best model to do prediction and Yolov9e.pt is the pre-trained model 



## Steps to Train the Model
1. Open train_demo.py file. 
2. Add the dataset directory path.(Ignore this if you want to use the default path)
2. Load the pre-trained yolov9e.pt model from the model folder.
3. Run the train function to train the model. 
4. It will store the validation resuls, detected results and the model in the run/detect/train folder.

Note - If training on cpu this process will take time so don't run for more than 1 epoch.

Use the best.pt model to predict results as it is trained properly with all the images the dataset here is just a sample dataset for you to try and train the model whole dataset link is provided above. The training for complete dataset requires GPU and takes more time to train.

## Testing 

There are 4 test in total.
1. Video_testing.py (mimicking the real scenario)
2. prediction.py (predict single image)
3. predict_multiple.py (predicts mutiple images)
4. train_demo.py (which contains the training code)

Use https://drive.google.com/file/d/1iNPShVs4savAWnnSpJplAdfEeTDbwecw/view?usp=share_link to download the video for video testing.