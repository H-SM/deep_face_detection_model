# import kivy dependencies

import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

#import kivy UX components
from kivy.uix.image import Image 
from kivy.uix.button import Button
from kivy.uix.label import Label

# import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Building our application layout 
class CamApp(App):

    def build(self):
        #Main layout 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_text = Label(text="Verification Uninitialized",  size_hint=(1,.1))

        # add items to layout 
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_text)

        # load keras model ( s-NN )
        self.model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={ 'L1Dist':L1Dist })

        # setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    #Run continously to get webcam feed
    def update(self, *args): 
        # Read frame from opencv 
        ret, frame = self.capture.read() 
        frame = frame[120:120+250,200:200+250 ,:]

        # Flip horizontal & canvert image to texture 
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # load image from file and convert to 100x100px 
    def preprocess (self, file_path): 
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100,100))
        img = img / 255.0
        return img

    #Bring over verification function to verify the person
    def verify(self, *args):
        # the thresholds
        detection_threshold = 0.8
        verification_threshold = 0.7

        # capture input image from webcam 
        save_path = os.path.join('application_data','input_image','input_image.jpg')
        ret, frame = self.capture.read() 
        frame = frame[120:120+250,200:200+250 ,:]
        cv2.imwrite(save_path, frame)

        #build result array 
        results = []
        for image in os.listdir(os.path.join('application_data','verification_images')):
            input_img = self.preprocess(os.path.join('application_data','input_image','input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data','verification_images',image))

            result = self.model.predict(list(np.expand_dims([input_img, validation_img],axis=1)))
            results.append(result)
    
        detection = np.sum(np.array(results)> detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data','verification_images')))
        verified = verification > verification_threshold

        # set verification text 
        self.verification_text.text = 'Verified' if verified == True else 'UnVerified'

        # log out details 
        Logger.info(results)
        Logger.info(verification)
        Logger.info(verified)
        Logger.info("IMAGE PASSED ->")
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.5))
        Logger.info(np.sum(np.array(results) > 0.8))
        Logger.info(np.sum(np.array(results) > 0.9))

        return results, verified

    
if __name__ == '__main__':
    CamApp().run()