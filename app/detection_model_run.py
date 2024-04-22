from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
from detectron2 import model_zoo

def build_config(weights_path):
    cfg = get_cfg()
    # Update the config with the model weights
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = weights_path 
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 16
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # person
    cfg.MODEL.RETINANET.NUM_CLASSES = 1

    return cfg


def run_detection(image_path, weights_path):
    cfg = build_config(weights_path)
    # Create the predictor
    predictor = DefaultPredictor(cfg)
    image = cv2.imread(image_path)
    # Perform prediction
    outputs = predictor(image)
    
    instances = outputs["instances"]
    #Get the index of the bounding box with the highest confidence score
    if len(instances) == 0:
        return 'no keypoints'
    else:
        highest_confidence_index = instances.scores.argmax()

    # Retrieve the keypoints associated with the highest confidence bounding box
    highest_confidence_keypoints = instances.pred_keypoints[highest_confidence_index].cpu().numpy()

    return highest_confidence_keypoints