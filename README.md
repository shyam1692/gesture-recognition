## Neural Networks for gesture-recognition
Contributors: Scott Mathews, Satendra Varma

In this project, we explore the application of Computer Vision in Gesture Recognition. 
Gesture Recognition is a field widely used in robotics.

We explore the recognition of limited number of gestures by a webcam with the person in the frame.

## Model Details
We have trained 2 neural network models:

# VGG16 Net:
We train VGG16 net using transfer learning and modify the final layers of the network.
This network is able to differentiate the gestures even in different lighting conditions, but due to the complexity of the network it can't be used for real time gesture recognition.

# Conventional Neural Network:
We train a smaller, conventional neural network. This network recognizes the gestures in real time, but doesn't work well in different lighting conditions.

# Future Scope
We can preprocess the data further (like converting the images to HSV instead of RGB) to generate more number of training examples and check the robustness to different lighting conditions.
We can also do object detection / localization of hands in the video frame, and use only the cropped image of hand for the purpose of classification, which would yield in higher accuracy.


