import torch, detectron2
import subprocess
import os
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import os
import numpy as np
from google.cloud import storage
import pandas as pd
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import copy
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader, build_detection_test_loader


def gcloud_initiation():
    try:
        # Run gcloud init command
        subprocess.run(["gcloud", "init"], check=True)
        print("gcloud init completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running gcloud init: {e}")

    # Upload the JSON key file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './service-account-key.json'

def grab_images():
    bucket_name = 'dl-individual-project'
    images_folder = 'trainImages/'
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs_train = bucket.list_blobs(prefix = images_folder)
    blob_names = [blob.name[len(images_folder):] for blob in blobs_train]

    #need to create the val and train folders before calling the two loops
    folder_paths = ["data/val", "data/train"]
    for folder_path in folder_paths:
        # Create the folder
        os.makedirs(folder_path, exist_ok=True)

    for name in blob_names:
        fullName = f"trainImages/{name}"
        tempName = f"data/train/{name}"
        blob = bucket.blob(fullName)
        blob.download_to_filename(tempName)

    images_folder = 'valImages/'
    blobs_val = bucket.list_blobs(prefix = images_folder)
    blob_names_val = [blob.name[len(images_folder):] for blob in blobs_val]

    for name in blob_names_val:
        fullName = f"valImages/{name}"
        tempName = f"data/val/{name}"
        blob = bucket.blob(fullName)
        blob.download_to_filename(tempName)
    
    #grab the detectron2 annotation files
    blob = bucket.blob('annotations/train_annotations.json')
    blob.download_to_filename('data/train_annotations.json')

    blob = bucket.blob('annotations/val_annotations.json')
    blob.download_to_filename('data/val_annotations.json')

   
def custom_mapper(dataset_dict):
  image = utils.read_image(dataset_dict["file_name"], format="BGR")
  transform_list = [
      T.Resize((800,600)),
      T.RandomLighting(0.5),
      T.RandomContrast(0.3, 0.5),  # Adjust the contrast range for dim lighting
      T.RandomSaturation(0.1, 0.3),
      T.RandomBrightness(0.1, 0.2),
  ]
  image, transforms = T.apply_transform_gens(transform_list, image)
  dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

  annos = [
      utils.transform_instance_annotations(obj, transforms, image.shape[:2])
      for obj in dataset_dict.pop("annotations")
      if obj.get("iscrowd", 0) == 0
  ]
  instances = utils.annotations_to_instances(annos, image.shape[:2])
  dataset_dict["instances"] = utils.filter_empty_instances(instances)
  return dataset_dict


class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

  @classmethod
  def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

  @classmethod
  def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=custom_mapper)
    
def main():
    gcloud_initiation()
    #grab_images()
    register_coco_instances("my_dataset_train", {}, "data/train_annotations.json", "data/train")
    register_coco_instances("my_dataset_val", {}, "data/val_annotations.json", "data/val")
    keypoint_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    keypoint_flip_map = [
        ('a', 'f'),  # 'r ankle' and 'l ankle'
        ('b', 'e'),  # 'r knee' and 'l knee'
        ('c', 'd'),  # 'r hip' and 'l hip'
        ('j', 'i'),  # 'head top' and 'upper neck'
        ('k', 'h'),  # 'r wrist' and 'l wrist'
        ('l', 'g'),  # 'r elbow' and 'l elbow'
        ('m', 'p'),  # 'r shoulder' and 'l shoulder'
    ]
    MetadataCatalog.get("my_dataset_train").thing_classes = ["person"]
    MetadataCatalog.get("my_dataset_train").thing_dataset_id_to_contiguous_id = {1:0}
    MetadataCatalog.get("my_dataset_train").keypoint_names = keypoint_names
    MetadataCatalog.get("my_dataset_train").keypoint_flip_map = keypoint_flip_map
    MetadataCatalog.get("my_dataset_train").evaluator_type="coco"
    cfg = get_cfg() # get a fresh new config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
    cfg.SOLVER.MAX_ITER = 800
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   #128   # faster (default: 512)
    #cfg.SOLVER.STEPS = (200, 500)
    cfg.SOLVER.GAMMA = 0.005
    cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((16, 1), dtype=float).tolist()

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # hand
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 16

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()


