# Simple face recognition using Python and OpenCV

This is a fairly simple code for facial recognition, using OpenCV2 and imutils

## Quick Start
**Installation**

`$ pip install opencv-python imutils`

**Start the code**

Make sure you have **deploy.prototxt.txt** file and the **Caffee Model** in your working directory 

`$ python detect_face.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_item_140000.caffemodel`

## FAQ
### Where is the feature video/gif?
Check the _last answer_!
### Why not use `cv2.VideoCapture(0)`?
For some reasons, _my cheap laptop webcams_ does not work very well with OpenCV
### Why don't you use a better webcam then?
Give me a job so I can afford a [**Panasonic Varicam 35**](https://na.panasonic.com/us/audio-video-solutions/broadcast-cinema-pro-video/cinema-cameras/varicam-35-4k-hdr-professional) for better resolution and detections
### Where is your face?
I am not 100% confident with myself to show my face on the internet yet, but you can find it at my [LinkedIn](https://www.linkedin.com/in/trunghaulelam/)
