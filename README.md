# Anime-Face-Detector
A Faster-RCNN based anime face detector.

This detector in trained on 6000 training samples and 641 testing samples, randomly selected from the dataset which is crawled from top 100 [pixiv daily ranking](https://www.pixiv.net/ranking.php?mode=daily).  

Thanks to [OpenCV based Anime face detector](https://github.com/nagadomi/lbpcascade_animeface) written by nagadomi, which helps labelling the data. 

The original implementation of Faster-RCNN using Tensorflow can be found [here](https://github.com/endernewton/tf-faster-rcnn)

## Dependencies
- Python 3.6.x
- tensorflow
- opencv-python
- cython
- Pre-trained ResNet101 model

## Usage
1. Clone this repository
    ```bash
    git clone https://github.com/qhgz2013/anime-face-detector.git
    ```
2. Download the pre-trained model  
    Google Drive: [here](https://drive.google.com/open?id=1WjBgfOUqp4sdRd9BHs4TkdH2EcBtV5ri)    
    Baidu Netdisk: [here](https://pan.baidu.com/s/1bvpCp1sbD7t9qnta8IhpmA)  
3. Unzip the model file into `model` directory
4. Build the CPU NMS model (skip this step if use PY_NMS with argument: `-nms-type PY_NMS`)
    ```bash
    make clean
    make
    ```
5. Run the demo as you want
    - Visualize the result (without output path):
        ```bash
        python main.py -i /path/to/image.jpg
        ```
    - Save results to a json file
        ```bash
        python main.py -i /path/to/image.jpg -o /path/to/output.json
        ```
        Sample output file:
        ```json
        {"/path/to/image.jpg": [{"score": 0.9999708, "bbox": [551.3375, 314.50253, 729.2599, 485.25674]}]}
        ```
    - Detecting a whole directory with recursion
        ```bash
        python main.py -i /path/to/dir -o /path/to/output.json
        ```
    - Customize threshold
        ```bash
        python main.py -i /path/to/image.jpg -nms 0.3 -conf 0.8
        ```
    - Customize model path
        ```bash
        python main.py -i /path/to/image.jpg -model /path/to/model.ckpt
        ```
    - Customize nms type (supports CPU_NMS and PY_NMS, not supports GPU_NMS because of the complicated build process for Windows platform)
        ```bash
        python main.py -i /path/to/image.jpg -nms-type PY_NMS
        ```

## Results
**Mean AP for this model: 0.9086**

![](./asset/sample1.png)
Copyright info: [東方まとめ](https://www.pixiv.net/member_illust.php?mode=medium&illust_id=54275439) by [羽々斬](https://www.pixiv.net/member.php?id=2179695)

![](./asset/sample2.png)
Copyright info: [【C94】桜と刀](https://www.pixiv.net/member_illust.php?mode=medium&illust_id=69797346) by [幻像黒兎](https://www.pixiv.net/member.php?id=4462245)

![](./asset/sample3.png)
Copyright info: [アイドルマスター　シンデレラガールズ](https://www.pixiv.net/member_illust.php?mode=medium&illust_id=69753772) by [我美蘭＠１日目 東A-40a](https://www.pixiv.net/member.php?id=2003931)
