import streamlit as st
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

import os
dir_path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(dir_path)

st.set_page_config(
    page_title="Visible Watermark",
    page_icon=":camera:",
    layout='wide'
)

st.title('📷 Visible Watermark')

st.subheader('Introduction')
st.markdown(
    """
    For demo purposes, we also insert a visible watermark, which is more of a traditional watermark. The pipeline is shown here:
    - Read the image
    - Random shuffle the position parameters using password
    - Simply add watermark using img + alpha * tmp, where alpha is the parameter to adjust the visibility of watermark
    Here is an example:
    """
)

st.markdown("\n")
st.markdown('\n')
st.markdown('\n')

st.subheader('Example for trial')
col1, col2, col3 = st.columns(3)
col1.write("Original Image")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
my_upload = col1.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        upload=my_upload
        ori = cv2.imread(my_upload)
else:
    upload = Image.open(dir_path+'/image outcome/visible watermark/ori.png')
    ori = cv2.imread(dir_path+'/image outcome/visible watermark/ori.png')

image = upload# Image.open(upload)
col1.image(image)

col2.write("Watermark")
my_upload2 = col2.file_uploader("Upload the watermark", type=["png", "jpg", "jpeg"])

if my_upload2 is not None:
    if my_upload2.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        upload2=my_upload2
        wm = cv2.imread(my_upload2)
else:
    upload2 = Image.open(dir_path+'/image outcome/visible watermark/esperantowatermark.png')
    wm = cv2.imread(dir_path+'/image outcome/visible watermark/esperantowatermark.png')

image2 = upload2# Image.open(upload)
col2.image(image2)

def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

col3.write("Outcome")

def EncodeVisible(ori, wm, alpha):
    img = ori
    height, width, channel = np.shape(img)
    watermark = wm
    wm_height, wm_width = watermark.shape[0], watermark.shape[1] #np.shape(watermark)
    x, y = list(range(int(height))), list(range(width))
    tmp = np.zeros(img.shape)
    for i in range(int(height)):
        for j in range(width):
            if x[i] < wm_height and y[j] < wm_width:
              tmp[i][j] = watermark[x[i]][y[j]]
    res = img + alpha * tmp # img_f + alpha * tmp
    res = np.real(res)
    cv2.imwrite('temporary.jpg', res)

# watermarked = Image.open('../Watermark/image outcome/visible watermark/ResWithPureWatermark.png')
value = col3.slider(
    'Select the parameter Alpha, which determines the visibility of the watermark',
    0, 100, 10)
EncodeVisible(ori, wm, value/100)
col3.download_button("Download watermarked image", open('temporary.jpg', 'rb').read(), "watermarked.jpg", "image/png")
col3.image('temporary.jpg')


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="JPG")
    byte_im = buf.getvalue()
    return byte_im