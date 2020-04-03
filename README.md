# Interview task
Here is my solution to interview task. I measured AP metric for two models performing on a part of FDDB dataset.

Some code was taken from here: 
https://github.com/opencv/opencv/blob/master/modules/dnn/misc/face_detector_accuracy.py
http://dlib.net/cnn_face_detector.py.html

I used <i>res10_300x300_ssd_iter_140000_fp16.caffemodel</i> in OpenCV that was downloaded using this script: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/download_weights.py

I also downloaded a model for dlib: http://dlib.net/files/mmod_human_face_detector.dat.bz2

To launch a program, use the following line:

<code>python3 face_detector_ap.py --cv_proto *path to OpenCV repo*/opencv/samples/dnn/face_detector/deploy.prototxt --cv_model *path to OpenCV repo*/opencv/samples/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel --dlib_model *path to a downloaded model for dlib*/mmod_human_face_detector.dat --fddb --ann *path to FDDB annotations*/FDDB-folds --pics *path to FDDB pictures*/originalPics</code>
