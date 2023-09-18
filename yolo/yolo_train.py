from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
def training_function():
    results = model.train(data='data.yaml', epochs=150, imgsz=640)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    
    training_function()