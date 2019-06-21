ICIP_YOLOv2_pytorch
=======

A implementation of YOLOv2 with pytorch for ICIP.

## Installation

Install dependency package

```
pip3 install -r requirements.txt
```

## Training
```
python3 train.py
```
## Architecture
```
-ICIP_YOLOv2_pytorch
    |- train.py // the main function
    |
    |- model.py // the class for total train step
    |
    |- lossfn.py // the class for lossfunction 
    |
    |- utils/
    |    |- anchor_box.py // calculate anchor box size with kmeans
    |    |
    |    |- dataset.py // the class to load data and preprocessing
    |    |
    |    |- resnet.py // the network(17 resnet layers + 7 additional layers)
    |    |
    |    |- transform.py // middleware to transform data format
    |    |
    |    |- util.py // utils function
    |    |
    -    |- show_img.py // show result in image 
    
```
## Configuration
You can set the parameters in the `config.json`
* train_image_folder
    * the folder path for training image
* train_annot_folder
    * the folder path for training annotation 
* pretrained_weights
    * the path for the pretrained weight
    * set null that will random initialization with part of resnet pretrained weight
* pretrained_kmean
    * the path for the pretrained kmeans
    * set null that will train a new kmeans model
* batch_size
    * batch_size for training
* learning_rate
    * the initialization of learning rate  
* epochs
    * epochs for training 
* step_size
    * the number of epoch to decay the learning rate
* decay_ratio
    * learning rate decay ratio
 
* saved_weight_name
    * the path for the new weight to save
* saved_kmean_name
    * the path for new kmeans to save

