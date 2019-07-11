ICIP_YOLOv2_pytorch
=======

A implementation of YOLOv2 with pytorch for ICIP.

## 題目介紹
* 比賽題目
    * 尋找病媒蚊孳生源-積水容器影像物件辨識
* 簡介
    * 藉由各種積水容器之標註資料，希望能透過影像物件偵測技術，能讓稽查人員藉由影像或是視訊提醒其積水容器之物件位置，除了提高稽查效率外，更期望進一步運用於其他載具中。
* 範例資料
    * ![](https://i.imgur.com/NWKSvzm.png)
## 實做架構
這次的比賽中，我們利用了Yolov2這個物件辨識的方法，並且實做了論文中所有的部件，包含DataLoader, Preprocessing, Network, LossFunction, NMS等。
### 架構
* Input Size : 448 * 448 * 3
* Output Size : 14 * 14 * (5 * 5 + 13)
    * 對於長寬分別切成14等分，每張圖片會被切成14 * 14個grid cells
    * 每個grid cell會對於13個classes進行預測，因此有13
    * 每個grid cell產生出5個bounding box的預測，分別預測w, h, x, y, c，因此為5 * 5
        * w, h為bounding box的長寬
        * x, y為bounding box的起始點
        * c為信心分數，表示這個bounding box和ground truth重疊了多少範圍
* 網路結構
    * 總共24層的CNN
        * 前面17層為Resnet19中的前17層pretrained layer
        * 後7層為random initialize的layers，透過stride和kernel size的調整讓output剛好為我們所需要的output size(14 * 14 * (5 * 5 +13))
    * ![](https://i.imgur.com/AmiWxKQ.png)
* Anchor box
    * 我們利用了k-mean來獲得在training dataset中最適合的anchor box
    * ![](https://i.imgur.com/o1ajUa6.png)
## 實做過程
### Weight Initialization
* 在前17層的Resnet layer中我們使用的是Resnet的pretrain weight
    * 因為我們認為即是是在object detection的task當中前面的layer也是在做feature extraction，所以利用了training在ImageNet的Resnet layer已經擁有了幾乎全世界所有feature(包含各種點線面)
* 而後7層的weight我們是用random initialization的方式
* 但由於下7層以random init的方式，因此在剛開始的epoch會產生與結果差距巨大的結果，導致loss和graient都很大，這樣的情況下，會將上17層pretrained weight也update至壞掉的方向，因此在training的過程，我們會先將上17層的gradient freeze住，當下面7層training到一個stable的情況下，我們才將整個network的gradient打開。 
### LossFunction
#### Paper中的lossfunction
這個是論文中，計算bounding box的長寬和起始位置的方式
* <img src="http://latex.codecogs.com/gif.latex?b_x = \sigma(t_x) + c_x" />
* <img src="http://latex.codecogs.com/gif.latex?b_y = \sigma(t_y) + c_y" />
    * <img src="http://latex.codecogs.com/gif.latex?b_x" /> 和 <img src="http://latex.codecogs.com/gif.latex?b_y" />為真正predict value，而<img src="http://latex.codecogs.com/gif.latex?t_x" /> 和 <img src="http://latex.codecogs.com/gif.latex?t_y" />為network predict出來的value
    * 在network predict出value之後會先經過sigmoid function並且加上cell的offset才是真正的位置
* <img src="http://latex.codecogs.com/gif.latex?b_w = p_w * e^{t_w}" />
* <img src="http://latex.codecogs.com/gif.latex?b_h = p_h * e^{t_h}" />
    * <img src="http://latex.codecogs.com/gif.latex?b_w" />  和 <img src="http://latex.codecogs.com/gif.latex?b_h" />為真正predict value，而<img src="http://latex.codecogs.com/gif.latex?t_w" /> 和 <img src="http://latex.codecogs.com/gif.latex?t_h" />為network predict出來的value，$p_w$ 和 $p_h$為anchor box的尺寸
    * 在network predict出value之後會先經過exp並且乘上anchor box的尺寸裁為真正的尺寸
#### Improve Lossfunction
* 依照paper中所提的lossfunction實做的過程中，我們發現雖然x, y可以達到良好的預測，但是w, h的結果卻很差，因此我們對於lossfunction做了一些改進，改進理由如下
* ![](https://i.imgur.com/dogoQQF.png)
    * 由於在predict的過程中xywh都是經過normalize的value(介於0～1之間)
    * 而對於經過sigmoid的xy，理想的值雖然是在-5～5之間，但實際上卻是可以是無窮的範圍內
    * 而對於經過exp的wh，則最好的範圍是在-1～1之間，一旦稍微超出便會導致loss過大，且gradient過大，便需要model自己學習將predict的值壓縮在-1～1之間
    * 對於這兩項，我們認為導致了整個model學習的方向不同，容易使model產生不穩定的情況，尤其是對於weight random initialization的情況下
* 對於wh進行改進後的lossfunction
    * <img src="http://latex.codecogs.com/gif.latex?b_w = p_w * e^{tanh(t_w)}" />
    * <img src="http://latex.codecogs.com/gif.latex?b_h = p_h * e^{tanh(t_h)}" />
* 這樣的改進可以讓model對於predict wh時，也如xy一樣理想範圍在-5～5之間，而實際上可以是無窮範圍內
* 並且我們得到的結果是w,h對於準確度高，且收斂速度比較快（因為tanh相當於進行了normalize） 

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

