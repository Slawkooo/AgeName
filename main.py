# import the opencv library
import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(r"C:\Users\slawo\PycharmProjects\AgeFace\model.h5")
def image_prep(image):
    image = cv2.resize(image, (224, 224))
    image = image.flatten()
    image = image / 255
    image = np.array(image)
    image = image.reshape(1, 150528)
    return image

x = cv2.imread('28775.jpg')
y = cv2.imread('28492.jpg')

x = image_prep(x)
y = image_prep(y)

prediction = model.predict(x)
# prediction = (prediction * 30) + 20
print(prediction)
prediction = model.predict(y)
print(prediction)

# define a video capture object
# vid = cv2.VideoCapture(0)
# iteracje  = 0
# while (True):
#
#     # Capture the video frame
#     # by frame
#     ret, frame = vid.read()
#     x = cv2.resize(frame, (224, 224))
#     x = x.flatten()
#     x = x / 255
#     print(x[0])
#     x = np.array(x)
#     x = x.reshape(1,150528)
#
#     prediction = model.predict(x)
#     prediction = (prediction * 30) + 20
#     print(prediction)
#     x1 = x
#     frame = cv2.resize(frame, (224, 224))
#     font = cv2.FONT_HERSHEY_SIMPLEX
#
#     # org
#     org = (50, 200)
#
#     # fontScale
#     fontScale = 0.5
#
#     # Blue color in BGR
#     color = (255, 0, 0)
#
#     # Line thickness of 2 px
#     thickness = 1
#     frame = cv2.resize(frame, (224, 224))
#     frame = cv2.putText(frame, 'Age: ' + str(prediction[0]), org, font,
#                        fontScale, color, thickness, cv2.LINE_AA)
#
#     # Display the resulting frame
#     x = cv2.resize(x, (224, 224))
#     cv2.imshow('frame', x)
#
#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()
