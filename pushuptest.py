# import the opencv library
import cv2
import numpy as np

import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# define a video capture object
vid = cv2.VideoCapture(0)

previous_total_pushups = None
current_total_pushups = -0.5

previous_pushup_state = None

while(True):
      
    # Capture the video frame by frame
    check, frame = vid.read()
    
    img = cv2.resize(frame,(224,224))

    test_image = np.array(img, dtype = np.float32)
    test_image = np.expand_dims(test_image,axis = 0)

    normalised_image = test_image/255.0

    # Prediction Result
    prediction = model.predict(normalised_image)
    #print("Prediction : ",prediction)
    
    if prediction[0][0] > 0.5:
        current_pushup_state = "up"
    else:
        current_pushup_state = "down"

    if current_pushup_state != previous_pushup_state:
        #print(current_pushup_state)
        previous_pushup_state = current_pushup_state
        current_total_pushups += 0.5

    if current_total_pushups != previous_total_pushups:
        #print(current_total_pushups)
        previous_total_pushups = current_total_pushups
    
    if current_total_pushups < 20:
        image = cv2.putText(frame, 'PushUp Count: ' + str(current_total_pushups), (10,50), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (0,0,0), 2, cv2.LINE_AA)
    else:
        image = cv2.putText(frame, 'CONGRATULATIONS 20 PUSHUPS', (10,50), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (0,0,0), 2, cv2.LINE_AA)        

    # Display the resulting frame
    cv2.imshow('frame', image)

    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()