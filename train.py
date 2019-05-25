import json
from utils.anchor_box import calculate_anchor, get_kmean, get_anchor
from utils.dataset import Detectionset

from model import DetectionModel

from torch.utils.data import DataLoader

CONFIG_PATH  = "./config.json"

if __name__ == "__main__":


    with open(CONFIG_PATH) as cb:
        config = json.loads(cb.read())

    #calculate_anchor(
    #            config["model"]["input_size"],
    #            config["train"]["train_annot_folder"],
    #            config["train"]["saved_kmean_name"]
    #        )
    kmean = get_kmean(config["train"]["pretrained_kmean"])
    anchor_box = get_anchor(kmean)

    data = Detectionset(
                kmean,
                config["classes"],
                config["train"]["train_image_folder"],
                config["train"]["train_annot_folder"],
                config["model"]["input_size"],
                config["model"]["S"],
                config["model"]["B"],
            )
    dataloader = DataLoader(
                data,
                batch_size=config["train"]["batch_size"],
                shuffle=True,
                num_workers=4
            )

    model = DetectionModel(
                dataloader,
                anchor_box,
                config["train"]["train_image_folder"],
                config["train"]["train_annot_folder"],
                config["train"]["pretrained_weights"],
                config["train"]["saved_weight_name"],
                config["train"]["batch_size"],
                config["train"]["object_scale"],
                config["train"]["no_object_scale"],
                config["train"]["coord_scale"],
                config["train"]["class_scale"]
            )
    model.train(
                config["train"]["epochs"],
                config["train"]["learning_rate"],
                config["train"]["step_size"],
                config["train"]["decay_ratio"]
            )
    
