import os
from ultralytics import YOLO

def main():
    
    model = YOLO('yolov8n.pt')

    
    yaml_path = "D:/hwa/github project/myanmar-car-license-plate/data.yaml"

    
    model.train(
        data=yaml_path,
        epochs=30,          
        imgsz=640,          
        batch=32,           
        device=0,           
        workers=4,          
        project="myanmar-car-plate", 
        name="yolov8_training"       
    )

if __name__ == '__main__':
    main()