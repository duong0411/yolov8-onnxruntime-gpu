import cv2
import yaml
from yolov8 import YOLOv8
import os
import time
if __name__ == "__main__":
    config_fp = "./configs/yolov8_onnx.yaml"
    input_dir = "Input data"
    output_dir = "Save data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
    with open(config_fp, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    total_time =0
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        imgs = cv2.imread(image_path)

        yolov8_onnx = YOLOv8(config)
        start = time.perf_counter()

        preds = yolov8_onnx.inference(imgs)
        elapsed_time = time.perf_counter() - start
        total_time += elapsed_time

        # for i, det in enumerate(preds):
        #     for box in det:
        #         cv2.rectangle(imgs, (int(box[0]), int(box[1])), (int(box[2]), int(box[3]), (255, 0, 255), 1))
        #         cv2.imwrite(output_path, imgs)

    print(f"Total inference time for {len(image_files)} images: {total_time:.2f} seconds")

