# RealTime-FaceReplacement

**This project proposes a method for real-time face replacement using a webcam. Most of the current researches on face replacement techniques are based on static face replacement techniques. Those studies focused on how to achieve face replacement between people in two photographs. This paper is devoted to achieving real-time face replacement based on the static face replacement technique. Ultimately, real-time face replacement technique has been implemented by merging the algorithms of face recognition, convex hull, Delaunay triangulation, and Poisson equation. In addition, there is also face colour contrast after real-time face replacement was performed. The colour difference between the central part of the new face and the original face was significant in comparison to the colour difference between the boundaries of both the new face and original face. The results also show that there are some limitations of the real-time face replacement function. The limitations include differences in face size, facial occlusions, and the rotation angle of the face in which they have a great impact on the outcome.**

Swap face between two face for Python 3 with OpenCV, numpy and dlib.

## Get Started
```sh
bash setup.sh
```
## Install 

### Requirement

* `pip install tensorflow`
* `pip install imutils`
* `pip install numpy`
* `pip install opencv-python`
* `pip install matplotlib`
* `pip install scipy`
* `pip install sklearn`
* `pip install keras`

Note: See [requirements.txt](requirements.txt) and [setup.sh](setup.sh) for more details.


### File Structure
```
.
├── README.md
├── face_detector
│    ├── deploy.prototxt
│    ├── res10_300x300_ssd_iter_140000.caffemodel
├── face_training
│    ├── dataset
|    │    ├── with_mask
|    |    │    ├── *.jpg
|    │    ├── without_mask
|    |    │    ├── *.jpg
│    ├── main.py
│    ├── mask_detector.model
│    ├── plot.png
├── main.py
├── requirements.txt
├── setup.sh
```
