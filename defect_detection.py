import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, model_name="model/yolov9n.pt", data_dir="data", test_dir="test_outputs"):
        self.model_name = model_name
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.data_config = None
        self.model = None
        self.classes = None


    def load_dataset(self, dataset_path):
        """Load dataset from directory 

        Args:
            dataset_path : path to the dataset containing images, labels and yaml 

        Raises:
            ValueError: If the dataset URL is invalid or the download fails.
        """
        try:
            self.data_config = self.validate_dataset(dataset_path)
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")



    def validate_dataset(self, data_dir):
        """
        Validates the dataset format for YOLOv9 based on the specified structure.

        Args:
            data_dir (str): Path to the root directory of the dataset.

        Raises:
            ValueError: If any validation check fails.

        Returns:
            returns the data_config path the data.yaml path for the model to train of it.
        """
        # check all the required folder are present in the folder
        required_folders = ['train', 'valid', 'test']
        data_config_path = Path(data_dir) / 'data.yaml'  # YAML file name is 'data.yaml'

        # Check if all required folders exist
        for folder in required_folders:
            if not os.path.exists(Path(data_dir) / folder):
                raise ValueError(f"Required folder '{folder}' not found in dataset directory.")

        # Check if YAML file exists
        if not os.path.exists(data_config_path):
            raise ValueError(f"Dataset configuration file 'data.yaml' not found.")

        # Check YAML format and required keys
        with open(data_config_path, 'r') as f:
            try:
                data_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML format: {e}")

        required_keys = ['train', 'val', 'test', 'nc', 'names']
        unexpected_keys = set(data_config.keys()) - set(required_keys)


        if unexpected_keys:
            raise ValueError(f"Unexpected keys found: {unexpected_keys}")
        
        missing_keys = set(required_keys) - set(data_config.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in YAML file: {', '.join(missing_keys)}")

        if not isinstance(data_config['nc'], int):
            raise ValueError(f"nc key must be an integer (number of classes).")

        # Validate image and label folder existence for train, val, test splits
       
        for split in ['train', 'valid', 'test']:
            image_dir = Path(data_dir) / split / 'images'
            label_dir = Path(data_dir) / split / 'labels'
            if not os.path.exists(image_dir):
                raise ValueError(f"Image directory not found: {image_dir}")

            if not os.path.exists(label_dir):
                raise ValueError(f"Label directory not found: {label_dir}")

            # Check for .txt files in label directory
            label_files = os.listdir(label_dir)
            if not all(file.endswith('.txt') for file in label_files):
                raise ValueError(f"Label directory ({label_dir}) must contain only .txt files.")
        
        return data_config_path
    


    def load_model(self, pretrained_model = "model/yolov9e.pt"):
        """Load model can be used to load the pre-trained model for model training
            and can be used after training to predict the results        
        """
        self.model = YOLO(pretrained_model)
        


    def train(self, epochs=1, batch=-1, imgsz= 640):
        """Train the yolo model on pre trained yolov9e model. 
        After training this function will store following components in the run/detect/train folder
        - weights 
            - Last.pt (latest trained model)
            - best.pt (best out of the trained model)
        - Different model validation graphs.
        - Different performance metrics in the excel file.
        - sample detected images.
        
        Args:
            epochs (int) : Number of epochs you want to use to train the model
            batch (int) : specify the batch size to train the model
            imgsz (int) : size of the image
            save_dir (str) : save the logs of the file at that path
        """
        self.model.train(data = self.data_config, epochs=epochs, batch=batch, imgsz=imgsz)
    

    def restart_training(self, saved_weights = 'runs/detect/train7/weights/last.pt'):
        """Start training the model from where you left at last. 

        Args:
            saved_weights - After training the model it will store the last state of
            the trained model in runs/detect/train/weights/last.pt This will be stored
            in the working directory.
        
        """
        self.model = YOLO(saved_weights)  # load a partially trained model

        # Resume training
        self.model.train(resume=True)
    
    def model_prediction(self, model, image_path):
        """Predict the values of the trained model. First load the model using load_model function
        """
        self.load_model(model)
        # model = YOLO(model)  # load a custom model
        # for file in os.listdir("/home/dssgis/ondemand/Yolov9_training/Dataset_split/test/images"):
        results = self.model.predict(image_path)
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            result.show()  # display to screen
    

    def predict_multiple(self, model, folder_path, result_path="results"):
        """Predict the images in the folder and store those images to results folder
        """

        self.load_model(model)
        for file in os.listdir(folder_path):
            extension = file.split(".")[-1]
            if extension not in ["jpg", "jpeg", "png"]:
                continue
            results = self.model.predict(os.path.join(folder_path,file))
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                result.save(filename=os.path.join("results",file))  # save to disk

            

if __name__ == "__main__":

    data_dir = os.path.join(os.path.dirname(__file__), "Dataset")
    od = ObjectDetection()

    try:
        # load the dataset 
        od.load_dataset(data_dir)
        #load the pre-trained model
        od.load_model("model/best.pt")
        #start training
        od.train()
        # od.model_prediction("model/best.pt", "Dataset/test/images/img_01_4402724300_00001_jpg.rf.7bcfacc21bccbec82f4e03e748484b35.jpg")
        # od.predict_multiple("model/best.pt", "Dataset/test/images")
        
    except ValueError as e:
        print(f"Dataset format validation error: {e}")

