#!/usr/bin/env python
# coding: utf-8



from ultralytics import YOLO 

model = YOLO("/home/dssgis/Training_segmentation/pre-trained/yolov8s-seg.pt")  # load a pretrained model (recommended for training)
model.info()

results = model.train(data=r"/home/dssgis/Training_segmentation/Dataset/Datasetv9_yolov8/data.yaml", epochs=100, imgsz=640)


# ## Validate Model

# In[ ]:



# Load a model
# model = YOLO("/home/dssgis/Training_segmentation/pre-trained/yolov8s-seg.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# # Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map  # map50-95(B)
# metrics.box.map50  # map50(B)
# metrics.box.map75  # map75(B)
# metrics.box.maps  # a list contains map50-95(B) of each category
# metrics.seg.map  # map50-95(M)
# metrics.seg.map50  # map50(M)
# metrics.seg.map75  # map75(M)
# metrics.seg.maps  # a list contains map50-95(M) of each category


# ## Model Prediction

# In[ ]:


# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolov8s-seg.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# # Predict with the model
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

