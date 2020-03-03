Tham khảo link colab sau:
https://drive.google.com/open?id=1akd6FWyBgSNF8ssyVbhNKVVYqtoiMqts

Tham khảo:
https://github.com/matterport/Mask_RCNN
https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46

Requirements:

    numpy==1.17.5
    scipy==1.0.0
    Pillow
    cython==0.29.15
    matplotlib==3.1.3
    scikit-image==0.16.2
    tensorflow>=1.3.0
    keras>=2.0.8
    opencv-python==4.2.0.32
    h5py
    imgaug
    IPython[all]


Các bước cài đặt trên môi trường window 10, anaconda 4.7.12 :

 - Sử dụng anaconda prompt (admin), tạo môi trường độc lập cho dự án: 

    conda create --name solardrone python=3.8.1
    conda config --add channels conda-forge
    conda install --no-channel-priority --file requirements-window.txt -y

Các bước cài đặt trên môi trường linux hoặc wsl:
    
    sudo apt install python3
    sudo apt install python3-pip
    sudo python3 -m pip install --upgrade pip
    git clone https://github.com/tucachmo2202/Solar_mark_rcnn.git
    python3 -m pip install setuptools --upgrade
    sudo apt-get install libsm6 libxext6 libxrender-dev
    python3 -m pip install -r requirements.txt
    mkdir /Solar_mark_rcnn/Mask_RCNN-2.1/logs/
    mkdir /Solar_mark_rcnn/Mask_RCNN-2.1/logs/solar/
    cd /Solar_mark_rcnn/Mask_RCNN-2.1/logs/solar/
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FgCQDJgMSEGJepDUagN2YOpLCaH7WXBN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FgCQDJgMSEGJepDUagN2YOpLCaH7WXBN" -O "mask_rcnn_solar_0050.h5" && rm -rf /tmp/cookies.txt
    cd /content/Solar_mark_rcnn/Mask_RCNN-2.1/samples/solar

Chạy mạng với những ảnh trong folder datasets/solar/val:

    python3 samples/solar/detect.py


Tool gán nhãn sử dụng: https://github.com/wkentaro/labelm
Tool gán nhãn sẽ xuất kết quả ra file .json
File ảnh phải có định dạng là .jpg hoặc .jpeg, copy tất cả file ảnh và file gán nhãn vào thư mục datasets/solar/train
Các lệnh để train:
    
    ```
    cd "/samples/solar"
    ```
    ```
    python3 solar.py train --dataset="/datasets/solar" --weights=coco
    ```
