# YOLO FACE + ID TRACKING

## Forked Project

This Project is a fork of [*yoloface*](https://github.com/sthanhng/yoloface).

## Pre-requisites

* python 3.7
* tensorflow, opencv, numpy

```
python -m pip install tensorflow opencv-python opencv-contrib-python numpy
git clone https://github.com/x2ever/Face-Tracking
cd Face-Tracking
```

## DownLoad Model Weight File

**Before starting, You shoud place these files into a directory `./model-weight/`**


#### CPU Model
    
DownLoad `yolov3-wider_16000.weights` file from [Google Drive Link](https://drive.google.com/file/d/1xYasjU52whXMLT5MtF7RCPQkV66993oR/view?usp=sharing)

#### GPU Model

DownLoad `YOLO_Face.h5` file from [Google Drive Link](https://docs.google.com/uc?export=download&id=1a_pbXPYNj7_Gi6OxUqNo_T23Dt_9CzOV)



## Run Model

#### CPU

> webcam

    python yoloface.py --src YOUR_WEBCAM_NUMBER(Mostly 0)

> video

    python yoloface.py --video PATH_TO_VIDEO_FILE 

#### GPU

> webcam

    python yoloface_gpu.py --src YOUR_WEBCAM_NUMBER(Mostly 0)

> video

    python yoloface_gpu.py --video PATH_TO_VIDEO_FILE 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for more details.


