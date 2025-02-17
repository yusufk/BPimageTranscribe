import json
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import PIL
from PIL import Image
from io import BytesIO
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

def adjust_gamma(image, gamma=1.0):
        """
        Credit: https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python
        Parameters:
            image: A grayscale image (NxM int array in [0, 255]
            gamma: A positive float. If gamma<1 the image is darken / if gamma>1 the image is enlighten / if gamma=1 nothing happens.
        Returns: the enlighten/darken version of image
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
        return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def get_lcd(image):
    image=np.array(image)
    #Preprocess image under test
    if image is not None:
        #Convert image to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Apply bilateral filter for smoothing test image(noise reduction), while preserving edges    
        blurred = cv2.bilateralFilter(image, 11, 11, 11)
        #Apply gamma correction to adjust image illumination
        gamma = adjust_gamma(blurred, gamma=0.7)
        #Apply adaptive thresholding with a mean neighborhood threshold calculation as image may have varying illuniation
        adt_thresh=cv2.adaptiveThreshold(gamma,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        #Erode thresholded image using a 3x3 rectangular kernel
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        eroded=cv2.erode(adt_thresh, kernel)
        #Invert image to find contours in image
        inverse=cv2.bitwise_not(eroded)
        contours, _ = cv2.findContours(inverse.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)   
        bp_cnt = []
        d=image.copy()  
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            cv2.rectangle(d,(x,y),(x+w,y+h),(255,0,0),2)
            aspect_ratio=w/h
            size=w*h
            if 0.95<= aspect_ratio<=1.5 and 20000<= size <80000:# and int(aspect_ratio)!=int(prevb):
                bp_cnt.append(contour)
                coord=int(str(x)[:2])
                prevb=aspect_ratio
        if bp_cnt!=[]:
            cnt2=max(bp_cnt, key=cv2.contourArea)
            x,y,w,h=cv2.boundingRect(cnt2)
            print("BP",x,y,w,h,w*h,w/h, cv2.contourArea(cnt2))
            upper_left = (int(w / 8), int(h / 16))
            bottom_right = (int(w * 7 / 8), int(h * 15 / 16))
            frame=inverse[y:y+h, x:x+w]
            mask=cv2.rectangle(frame.copy(), upper_left, bottom_right, (0, 0, 0),-1)
            final_img=frame-mask    
            return final_img
    else: 
        print("Image not found")
        return None


def DataPrepImage(rawimage):
    preprocessed_img = get_lcd(rawimage)
    w, h=preprocessed_img.shape
    top =  preprocessed_img[0:int(h/2),0:w]
    top = Image.fromarray(top)
    top = top.resize((180,80))
    top = np.array(top)
    top = top.reshape((top.shape[0],top.shape[1],1))
    top_array = np.asarray([top])
    sys_pred = network.predict(top_array)
    for i in range(top_array.shape[0]):
        pred_list_i = [np.argmax(pred[i]) for pred in sys_pred]
        predicted_sys = 100* pred_list_i[0] + 10 * pred_list_i[1] + 1* pred_list_i[2]
        if predicted_sys >= 1000:
            predicted_sys = predicted_sys-1000
    bottom = preprocessed_img[int(h/2):h,0:w]
    bottom = Image.fromarray(bottom)
    bottom = bottom.resize((180,80))
    bottom = np.array(bottom)
    bottom = bottom.reshape((bottom.shape[0],bottom.shape[1],1))
    bottom_array = np.asarray([bottom])
    dia_pred = network.predict(bottom_array)
    for i in range(bottom_array.shape[0]):
        pred_list_i = [np.argmax(pred[i]) for pred in dia_pred]
        predicted_dia = 100* pred_list_i[0] + 10 * pred_list_i[1] + 1* pred_list_i[2]
        if predicted_dia >= 1000:
            predicted_dia = predicted_dia-1000
    return predicted_sys, predicted_dia

def init():
    global network    

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    folder = os.getenv('AZUREML_MODEL_DIR')
    if (folder==None):
        folder = "."

    model_path = os.path.join(folder, 'best_model.h5')

    #On charge le model Keras
    network = load_model(model_path)

@rawhttp
def run(request):
    if request.method == 'POST':
        reqBody = request.get_data(False)
        myImage = PIL.Image.open(BytesIO(reqBody))
        #Convert image to grayscale
        myImage = myImage.convert('L')

        #Prep and predict
        sys_pred, dia_pred = DataPrepImage(myImage)


        return {"sys": str(sys_pred), "dia": str(dia_pred)}
    else:
        return AMLResponse("bad request, use POST", 500)