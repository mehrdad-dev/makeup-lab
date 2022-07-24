import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import unet
import os
from skimage.transform import resize, rescale
from moviepy.editor import ImageSequenceClip
import base64

# ================================================================================================

HAIR_MODEL_PATH = 'hair-seg-12.hdf5'
LIP_MODEL_PATH = '/home/mehrdad/Documents/SELF-PROJECT/Makeup Lab/model'

# ================================================================================================

def resize_image(img, size=(224,224)):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1
    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w
    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_AREA

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2
    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


def transfer(clr, alpha=1.0):
    mask = cv2.imread('mask.jpg')
    image = cv2.imread('image.jpg')

    if clr=='red':
      color = [0,0,255]
    elif clr=='cyan':
      color= [255, 255, 0] 
    elif clr=='gold':
      color = [0, 255, 255]
    elif clr=='creamy':
      color = [255,255,255]           
    elif clr=='purple':
      color = [255,0,0]           
    elif clr=='green':
      color = [100, 225, 0]
    elif clr=='brown':
      color = [0, 0, 100]    # img = cv2.resize(img, (height, width), interpolation = cv2.INTER_NEAREST)
    elif clr=='blond':
      color = [0,100,160]   
    elif clr=='purple 2':
      color = [100,0,160]     
    elif clr=='navy blue':
      color = [200,50,10]  
    elif clr=='orange':
      color = [10,50,250]       
    elif clr=='green 2':
      color = [10,250,200]

    _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    green_hair = np.copy(image)
    green_hair[(mask==255).all(-1)] = color
    alpha = 0.7
    green_hair_w = cv2.addWeighted(green_hair, 1 - alpha, image, alpha, 0, green_hair)
    result = cv2.cvtColor(green_hair_w, cv2.COLOR_BGR2RGB)

    return result


def predict(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    img = img.astype('float64')
    a, b, _ = img.shape
    w, h = 224, int((b * 224 / a))
    img = resize(img, (w, h),mode='wrap', anti_aliasing=True)    
    img = resize_image(img)    
#    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
#    img = img[:,:, :3]
    im = img.reshape((1,) + img.shape)

    pred = model.predict(im)
    mask = pred.reshape((224, 224))
    plt.imsave('mask.jpg', mask, cmap='gray')
    plt.imsave('image.jpg', img)


@st.experimental_singleton
def load_model(model_path):
  model = unet.unet(224,224,1,3)
  model.load_weights(model_path)    
  # model = tf.keras.models.load_model(model_path)
  return model



def pipeline(model, image, video, color):
    if image is not None:
        predict(image, model)
        image_result = transfer(color, alpha=1.0)
        st.info('Your makeup is ready!')
        st.image(image_result, channels="RGB", caption='Your uploaded image')      
    else:
        results_frames = []
        while video.isOpened():
            if len(results_frames) >=175:
                break
            ret, frame = video.read()
            if not ret:
                st.info("End of video stream ...")
                break
            predict(frame, model)
            frame_result = transfer(color, alpha=1.0)
            results_frames.append(frame_result)

        clip = ImageSequenceClip(list(results_frames), fps=25)
        clip.write_gif('test.gif', fps=25)     
        st.info('It is ready! (wait to download it)')
        st.info('Memory is limited in the streamlit, so we only generate the first 7 seconds of your video!')
        file_ = open("test.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="gif">',
            unsafe_allow_html=True,
        )        
        # writer = cv2.VideoWriter('001.avi',-1, 25, (224,224))
        # # writer = cv2.VideoWriter('test1.avi', cv2.VideoWriter_fourcc(*'mp4'), 25, (224, 224), False)
        # for i in results_frames:
        #     writer.write(i)
        # writer.release()  


# ================================================================================================
st.title('Makeup Lab 🧑‍🔬💄')
st.markdown(
    'By [Mehrdad Mohammadian](https://mehrdad-dev.github.io)', unsafe_allow_html=True)

about = """
Apply different hair/lipstick color!

hair:  available

lipstick: as soon as possible
"""
st.markdown(about, unsafe_allow_html=True)


# ================================================================================================
#makeup_type = st.selectbox(
#     'Makeup for hair color or lipstick color?',
#     ('hair', 'lipstick'))
#
#
# ================================================================================================
file_type = st.selectbox(
     'Your file is video or image?',
     ('image', 'video'))

# ================================================================================================
image = None
video = None

if file_type == 'image':
    uploaded_image = st.file_uploader("Upload a jpg image", type=["jpg"])
    if uploaded_image is not None:
        # file_details = {"Filename":uploaded_image.name,"FileType":uploaded_image.type,"FileSize":uploaded_image.size}
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR", caption='Your uploaded image')
        
else:
    uploaded_video = st.file_uploader("Upload a mp4 video", type=["mp4"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        video = cv2.VideoCapture(tfile.name)

# ================================================================================================
COLOR = st.selectbox(
     'Select a color',
     ('red', 'cyan', 'gold', 'creamy', 'purple', 'purple 2', 'green', 'green 2', 'brown', 'blond',
     'navy blue', 'orange'))


# ================================================================================================


left_column, right_column = st.columns(2)
pressed = left_column.button('Predict!')
if pressed:
#  if makeup_type == 'hair':
#    model = load_model(HAIR_MODEL_PATH)
#  else:
#    model = load_model(LIP_MODEL_PATH)
  model = load_model(HAIR_MODEL_PATH)
  st.info('Model loaded!, please wait!')
  pipeline(model, image, video, COLOR)
  st.balloons()
