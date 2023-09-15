import streamlit as st
from PIL import Image
from io import BytesIO
import os
dir_path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(dir_path)

import blind_watermark
from blind_watermark import WaterMark
from blind_watermark import att
from blind_watermark.recover import estimate_crop_parameters, recover_crop
import cv2
import numpy as np

from fuzzywuzzy import fuzz

st.set_page_config(
    page_title="Invisible Watermark and Robustness Tests",
    page_icon=":bright_button:",
    layout='wide'
)

st.title('🔆 Invisible Watermark & Robustness Tests')

st.header('Simple Introduction')
st.markdown(
    """
    The main focus for this watermark tool is to gain robustness to attack by inserting information into the frequency domain space of the picture.
    \n
    After reading the image, we used the cvtColor package provided by cv2, transferring the image into YUV color space. For watermark, here is the pipeline for our invisible watermark:
    \n
    DCT transfer – (flatten – password shuffle – reverse flatten) – SVD – adding watermark – reverse SVD – (flatten – password reverse shuffle – reverse flatten) – reverse DCT
    The basic idea comes from (The reason using DCT + SVD)

    """
)
st.markdown("\n")
st.markdown('\n')
st.markdown('\n')
st.info("""
            👇 If you are interested, please click on the expander for more technical details. 👇
            """)

with st.expander("🖋️  ***Detailed Introduction of Inserting Watermark***", expanded = True):
    st.subheader("**Concept Introduction:**")
    st.markdown(
        """
        \n
        We also use another image watermarking method based on discrete wavelet transformation (DWT), Hessenberg decomposition (HD), and singular value decomposition (SVD). Through experiments, we find that the embedded watermark can be successfully extracted even after performing discrete wavelet transformation combined with some other attack. The method also provides a good trade-off between robustness and invisibility. 
        \n
        We first define the main decompositions and transformations utilized in this work:  
        \n
        ***DWT***:
        \n 
        - DWT is a technique that transforms image pixels into wavelets, which are then used for wavelet-based compression and coding. The DWT decomposes a digital signal into different subbands so that the lower frequency subbands have finer frequency resolution and coarser time resolution compared to the higher frequency subbands.
        \n
        - In our work, the original input image is transformed into four sub-bands, including low-high (LH), high-low (HL), high-high (HH), and low-low (LL). Most of the information contained in the original image is concentrated into the LL sub-band after one level of DWT. Numerous research have showed that LL has a better performance on the attacks, which makes it a good candidate for robust watermarking. 
        \n
        ***Hessenberg Decomposition***:
        \n
        - HD is used for square matrix decomposition. Research have shown the HD improves the robustness against the attacks by finding a more precise component of the host image. Assuming we have $n$ x $n$ dimensional matrix X, then HD($X$) = $PHP^T$, where *P* is an orthogonal matrix, *H* is an upper Hessenberg matrix and $h_{i,j} = 0$ when $i>j+1$. 
        \n
        ***Singular Value Decomposition***: 
        \n
        - SVD decomposes a symmetric matrix into three sub-matrices such that singular values get separated in the form of a diagonal matrix. 
        \n
        - Suppose *Y* is a symmetric matrix, then $SVD(Y) = USV^T$. The three decomposed matrices are the left singular matrix *U*, the singular matrix  S, and the right singular matrix V. The columns of *U* are orthonormal eigenvectors of $YY^T$, the columns of V are orthonormal eigenvectors of $Y^TY$, and *S* is a diagonal matrix that contains the square roots of the eigenvalues from *U* or *V* in descending order. 
        \n
        """
    )
    st.subheader("**Watermark Embedding:**")
    st.markdown(
        """
        \n
        The watermarking embedding procedure is shown in the following figure. In particular, the inputs are a host image (*C*), with dimension *M* x *M*; a watermark image (*W*), with dimension *N* x *N*; and the output is the watermarked host image *C*, with dimension  *M* x *M*.
        \n
        """
    )
    st.image(Image.open(dir_path+'/image outcome/DWT-1.png'))
    st.markdown(
        """
        \n
        The detailed steps of the embedding algorithm are as follows:
        \n
        """
    )
    st.markdown(r"""
        - 1) Decompose C into the components *LL*, *LH*, *HL*, *HH* based on R-level DWT, where R = $log_{2} \frac{M}{N}$.
        """)
    st.markdown("""
        \n  
        - 2) Perform Hessenberg Decomposition on *LL*: HD(*LL*) = $PHP^T$. 
        \n
        - 3) Apply SVD to H: SVD(*H*) = $HU_{w} HS_{w} HV_{w}^T$. 
        \n
        - 4) Apply SVD to W: SVD(*W*) = $U_{w} S_{w} V_{w} ^T$. 
        \n""")
    st.markdown(r"""
        - 5) Compute an embedded singular value $HS_{w}^{*}$: $HS_{w}^{*} = HS_{w} + \alpha S_{w}$, where $\alpha$ is a scaling factor. In our work, we fix $\alpha$ to be 0.1. 
        """)
    st.markdown("""
        \n
        - 6) Generate the watermarked sub-band $H^*$ by using inverse SVD: $H^* = HU_{w} HS_{w}^{*} HV_{w}^{T}$. 
        \n
        - 7) Reconstruct a new low-frequency approximate sub-band $LL^{*}$ based on the inverse HD: $LL^{*} = PH^{*}P^{T}$. 
        \n
        - 8) Obtain the watermarked image $C^{*}$ by performing the inverse R-level DWT. 
        \n
        """
    )
    st.subheader("**Watermark Extraction:**")
    st.markdown(
        """
        \n
        The watermarking extraction procedure is shown in the following figure. The input is the watermarked host image $C^{*}$, and the output is the extracted watermark $W^{*}$, with dimensions *N* x *N*.
        \n
        """)
    st.image(Image.open(dir_path + '/image outcome/DWT-2.png'))
    st.markdown("""
        \n
        The detailed steps of the watermark extraction algorithm are as follows:
        \n
        - 1) Decompose the watermarked host image $C^{*}$ into four sub-bands by R-level DWT. We denote the sub-bands as $LL_w, LH_w, HL_w, HH_w$. 
        \n
        - 2) Perform Hessenburg decomposition on $LL_w$ by $HD(LL_w)$ = $P_w H_w P_{w}^T$. 
        \n
        - 3) Apply SVD to $H_w$: $SVD (H_w) = HU_{w}^{*} HSb_{w}^{*} HV_{w}^{*^{T}}$.
        \n
    """)
    st.markdown(r"""
        - 4) Obtain the extracted singular value $S_{w}^{*}: S_{w}^{*} = \frac{(HSb_{w}^{*} - HS_{w}^{*})}{\alpha}$.
    """)
    st.markdown("""
        \n
        - 5) Reconstruct the extracted watermark $W^{*}$ by inverse SVD: $W^{*} = U_{w2} S_{w}^{*} V_{w2}^{T}$, where the two components decrypted by the chaotic system are marked as $U_{w2}$ and $V_{w2}^{T}$.
    """)
    st.subheader("**Robustness and Invisibility Measurement**")
    st.markdown("""
        \n
        We measure the invisibility of the watermark by the Peak Signal to Noise Ratio (PSNR) and the Structural Similarity Index Measure (SSIM):
        \n
        """)
    st.markdown(r"""
        PSNR(*C*,*C$^{*}$*) = 10 log $\frac{C_{max}^{2}}{MSE})$, where
        """)
    st.markdown("""
        \n
        - MSE is the mean square error between the host image and the watermarked host image,
        \n
        - $C_{max}$ is the maximum pixel value in the host image. 
        \n
        """)
    st.markdown(r"""
        $SSIM(C,C^*) = \frac{\mu_{C} \mu_{C^{*}}+d_{1}}{\mu_{C}^{2}+\mu_{C^{*}}^{2}+d_1} \frac{\sigma_{CC^{*}} +d_{2}}{\sigma_{C}^{2}+\sigma_{C^{*}}^{2}+d_2}$, where
        """)
    st.markdown("""
        \n
        - $\mu_{C}$ and $\mu_{C^{*}}$ are the average of $C$ and $C^*$,
        \n
        """)
    st.markdown(r"""
        - $\sigma_{C}$ and $\sigma_{C^{*}}$ are the variance of $C$ and $C^*$, $\sigma_{CC^{*}}$ is the covariance between $C$ and $C^*$, 
        """)
    st.markdown("""
        \n
        - and $d_1$ and $d_2$ are two variables which are used to stabilize the division with a weak denominator. 
        \n
    """)


st.markdown("\n")
st.markdown('\n')
st.markdown('\n')
st.header('Example for trial: Without Any Attack')

col1, col2, col3 = st.columns(3)
col1.write("Original Image")
col1.write("\n")
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
my_upload = col1.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        upload=my_upload
else:
    upload = Image.open(dir_path+'/image outcome/invisible watermark/NewYork.jpg')

image = upload# Image.open(upload)
img = image.save("ori_img.jpg")
ori_img = cv2.imread("ori_img.jpg")

col1.write("\n")
col1.image(image)

col2.write("Watermark")
col2.write("\n")
col2.write("\n")
col2.markdown(
"""
    You can try with your own secret code - type in the watermark you want to insert:
    """
)
wm = col2.text_input('watermark: ', 'Esperanto Technologies')
col2.markdown(
    """
    You can also set your own password for this watermark, so people without this password won't be able to extract it:
    """
)
password = int(col2.text_input('password: ', '350'))


# def convert_image(img):
#     buf = BytesIO()
#    img.save(buf, format="PNG")
#    byte_im = buf.getvalue()
#    return byte_im

col3.write("Outcome")

blind_watermark.blind_watermark.bw_notes.close()

os.chdir(os.path.dirname(__file__))
ori_img_shape = ori_img.shape[:2]  # 抗攻击有时需要知道原图的shape
h, w = ori_img_shape
# %% embed string into image whose format is numpy.array
bwm = WaterMark(password_img=1, password_wm=password)
bwm.read_img(img=ori_img)

bwm.read_wm(wm, mode='str')
watermarked = bwm.embed()
cv2.imwrite('watermarked.jpg', watermarked)

len_wm = len(bwm.wm_bit)  # 解水印需要用到长度

# watermarked = Image.open('../Watermark/image outcome/invisible watermark/NY.png')
password_ex = int(col3.text_input('password for extraction: ', '350'))

bwm1 = WaterMark(password_img=1, password_wm=password_ex)
wm_extract = bwm1.extract(embed_img=watermarked, wm_shape=len_wm, mode='str')

col3.write("The extracted Information is: "+ "\n"+ "***"+ wm_extract+ "***")
col3.write("The Similarity Score: "+ "***" + str(fuzz.ratio(wm, wm_extract)) + "***")
col3.image('watermarked.jpg')
col3.download_button("Download watermarked image", open('watermarked.jpg', 'rb').read(), "watermarked.png", "image/png")

st.markdown("\n")
st.markdown('\n')
st.markdown('\n')



st.header('Robustness Test')
st.markdown("""
    One of the most important criterion of watermark is ***Robustness***. We have following attacks for trial:
""")
st.subheader('Without Reverse Fuction: Pure Attack&Extract')
option = st.selectbox(
    'Which Attack you would apply to the watermarked picture?',
    ('Cut',
     'Scale',
     'Cut + Scale',
     'Rotate',
     'Block',
     'Salt & Pepper Noise',
     'Adjust Brightness',
     'Warp',
     'Color Rotation',
     'Fourier Attack'))

parameter_dict = {'Cut':'loc = ((x1= 0.3, y1= 0.1), (x2 = 0.7, y2 = 0.9) means the coordinates of the cutting attack',
               'Scale':'out_shape = (400, 300) means that the size of the output image is changed into (400,300)',
               'Cut + Scale': 'loc = ((0.1, 0.1), (0.5, 0.5)) means the coordinates of the cutting attack,' +'\n' + 'scale = 0.7 means the size of the image is 70% of the original image',
               'Rotate':'angle = 60 means rotate the image 60 degrees clockwise',
               'Block':'n = 60 means there are 60 small blocks',
               'Salt & Pepper Noise':'ratio = 0.05 means the ratio of the noise',
               'Adjust Brightness':'ratio=0.8 means the output image is darker than the original.' +'\n' + 'ratio = 1.1 means the output is brighter than the original image' +'\n' + 'It should be smaller than 1.5',
               # 'Warp':"",
               # 'Color Rotation':"",
               'Fourier Attack':'frequency = 25 means the frequency of the filtering'
}


st.write('You selected: ', option)
st.info(
    'For the attack '+ option + ' you selected, you can adjust the attacking parameters.' +'\n' + 'For example, '+ parameter_dict.get(option)+'.',
    icon="👾",
)

if option == 'Cut':
    coordinate = st.text_input('Please input your Attacking parameters(if any): loc = ', ((0.1, 0.2), (0.5, 0.5)))
    coordinate = coordinate.replace(" ", "")
    coordinate = coordinate.replace("(", "")
    coordinate = coordinate.replace(")", "")
    x = coordinate.split(",")
    password_1 = int(st.text_input('Please input your password for extracting the watermark:', '350'))
    x1, y1, x2, y2 = int(w * float(x[0])), int(h * float(x[1])), int(w * float(x[2])), int(h * float(x[3]))
    Attacked = att.cut_att3(input_img=watermarked, loc=(x1, y1, x2, y2), scale=None)
    cv2.imwrite('Attacked_'+option+'.jpg', Attacked)
    bwm1 = WaterMark(password_wm=password_1, password_img=1)
    wm_extract = bwm1.extract(embed_img=Attacked, wm_shape=len_wm, mode='str')
    simi_score = fuzz.ratio(wm, wm_extract)
elif option == 'Scale':
    coordinate = st.text_input('Please input your Attacking parameters(if any): shape = ', (400, 300))
    coordinate = coordinate.replace(" ", "")
    coordinate = coordinate.replace("(", "")
    coordinate = coordinate.replace(")", "")
    x = coordinate.split(",")
    password_1 = int(st.text_input('Please input your password for extracting the watermark:', password))
    x1, y1 = int(x[0]), int(x[1])
    Attacked = att.resize_att(input_img=watermarked, out_shape=(x1,y1))
    cv2.imwrite('Attacked_'+option+'.jpg', Attacked)
    bwm1 = WaterMark(password_wm=password_1, password_img=1)
    wm_extract = bwm1.extract(embed_img=Attacked, wm_shape=len_wm, mode='str')
    simi_score = fuzz.ratio(wm,wm_extract)
elif option == 'Cut + Scale':
    coordinate = st.text_input('Please input your Attacking parameters(if any): coordinate = ', ((0.1, 0.2), (0.5, 0.5)))
    size = st.text_input('Please input your Attacking parameters(if any): size = ', 0.6)
    coordinate = coordinate.replace(" ", "")
    coordinate = coordinate.replace("(", "")
    coordinate = coordinate.replace(")", "")
    x = coordinate.split(",")
    password_1 = int(st.text_input('Please input your password for extracting the watermark:', password))
    x1, y1, x2, y2 = int(w * float(x[0])), int(h * float(x[1])), int(w * float(x[2])), int(h * float(x[3]))
    Attacked = att.cut_att3(input_img=watermarked, output_file_name=None, loc=(x1, y1, x2, y2), scale=float(size))
    cv2.imwrite('Attacked_'+option+'.jpg', Attacked)
    bwm1 = WaterMark(password_wm=password_1, password_img=1)
    wm_extract = bwm1.extract(embed_img=Attacked, wm_shape=len_wm, mode='str')
    simi_score = fuzz.ratio(wm, wm_extract)
elif option == 'Rotate':
    angle = float(st.text_input('Please input your Attacking parameters(if any): angle = ', 60))
    password_1 = int(st.text_input('Please input your password for extracting the watermark:', password))
    Attacked = att.rot_att(input_img=watermarked, angle=angle)
    cv2.imwrite('Attacked_'+option+'.jpg', Attacked)
    bwm1 = WaterMark(password_wm=password_1, password_img=1)
    wm_extract = bwm1.extract(embed_img=Attacked, wm_shape=len_wm, mode='str')
    simi_score = fuzz.ratio(wm,wm_extract)
elif option == 'Block':
    n = int(st.text_input('Please input your Attacking parameters(if any): n = ', 30))
    password_1 = int(st.text_input('Please input your password for extracting the watermark:', password))
    Attacked = att.shelter_att(input_img=watermarked, ratio=0.1, n=n)
    cv2.imwrite('Attacked_'+option+'.jpg', Attacked)
    bwm1 = WaterMark(password_wm=password_1, password_img=1)
    wm_extract = bwm1.extract(embed_img=Attacked, wm_shape=len_wm, mode='str')
    simi_score = fuzz.ratio(wm,wm_extract)
elif option == 'Salt & Pepper Noise':
    ratio = float(st.text_input('Please input your Attacking parameters(if any): ratio = ', 0.05))
    password_1 = int(st.text_input('Please input your password for extracting the watermark:', password))
    Attacked = att.salt_pepper_att(input_img=watermarked, ratio=ratio)
    cv2.imwrite('Attacked_'+option+'.jpg', Attacked)
    bwm1 = WaterMark(password_wm=password_1, password_img=1)
    wm_extract = bwm1.extract(embed_img=Attacked, wm_shape=len_wm, mode='str')
    simi_score = fuzz.ratio(wm,wm_extract)
elif option == 'Adjust Brightness':
    ratio = float(st.text_input('Please input your Attacking parameters(if any): ratio = ', 0.9))
    password_1 = int(st.text_input('Please input your password for extracting the watermark:', password))
    Attacked = att.bright_att(input_img=watermarked, ratio=ratio)
    cv2.imwrite('Attacked_'+option+'.jpg', Attacked)
    bwm1 = WaterMark(password_wm=password_1, password_img=1)
    wm_extract = bwm1.extract(embed_img=Attacked, wm_shape=len_wm, mode='str')
    simi_score = fuzz.ratio(wm, wm_extract)
elif option == 'Fourier Attack':
    threshold = float(st.text_input('Please input your Attacking parameters(if any): threshold = ', 25))
    password_1 = int(st.text_input('Please input your password for extracting the watermark:', password))
    Attacked, filtered = att.fft_att2(image=watermarked, threshold=25)
    Attacked1 = att.resize_att(input_img=watermarked, out_shape=(w -int(threshold),h-int(threshold) * 2))
    cv2.imwrite('Attacked_'+option+'.jpg', Attacked)
    bwm1 = WaterMark(password_wm=password_1, password_img=1)
    wm_extract = bwm1.extract(embed_img=Attacked1, wm_shape=len_wm, mode='str')
    simi_score = fuzz.ratio(wm, wm_extract)

col1, col2 = st.columns(2)
col1.write("Original Image")
col1.write("\n")
col1.image('watermarked.jpg')

col2.write("Attacked Image")
col2.write("\n")

# Attacked = Image.open('../image outcome/invisible watermark/'+ option +'.png')
col2.image('Attacked_'+option+'.jpg')
col2.write("The extracted Information is: "+ "\n"+ "***" + wm_extract + "***")
col2.write("The Similarity Score: " + "***" + str(simi_score) + "***")


st.markdown("\n")
st.markdown('\n')
st.markdown('\n')
st.subheader('With Reverse Function: Reverse&Extract')
st.markdown("""
    \n
    You may have noticed that for several attacks, if we don't know the attacking parameters, the similarity score between extracted watermark and original watermark will be fairly low.
    \n
    Thus, we tried to add anti-attack functions for several attacks, like cutting, scaling, fourier attack, etc.
    \n""")
st.info("""
            👇 If you are interested, please click on the expander for more technical details. 👇
            """)

with st.expander("🖋️  ***Detailed Introduction of Reverse Attack***", expanded = True):
    st.markdown("""
        - ***Cutting***: we use the **matchTemplate()** function from opencv.
            - It is a method for searching and finding the location of a template image in a larger image. 
            - It simply slides the template image over the input image (as in 2D convolution) and compares the template and patch of input image under the template image. 
            - It implemented several comparison methods.
            - If input image is of size (WxH) and template image is of size (wxh), output image will have a size of (W-w+1, H-h+1). 
            - After that, we just use cv.minMaxLoc() function to find where is the maximum/minimum value. Take it as the top-left corner of rectangle and take (w,h) as width and height of the rectangle. 
            \n
            The rectangle is your region of template and now we successfully estimate the attacking parameters of Cutting.
        \n
        - ***Scaling***: we create a search function for the parameter scale.
            - We first calculate the min/max scale.
            - We get the w and h by multiply the attacked pictures shape by *scale*.
            - Then, we resize the attacked image and get the matching score by using the **matchTemplate()** function from opencv. Save it temporarily as tmp_score.
            - If the new matching score is higher then tmp_score, then tmp_score = current score and we update the min/max scale.
            \n
            Stop until we get the highest score's corresponding scale, and now we successfully estimate the attacking parameters of Scaling.
        \n
        - ***Fourier Attack***: we create a search function for the parameter window size.
            - The filter we've been using has a parameter window size, which can be a trial for parameter searching.
            - We set the min/max window size, and reverse the attacked image with different window size.
            - With the reversed images, we applied the *extract_watermark* function, and calculate the similarity score.
            - We assume the parameter with the highest similarity score is the attacking parameter.
            \n
            We tried to simplify it using dichotomy, however, there're some cases that violates increasing or decreasing relationship. Thus, we decided to do the whole searching.
        \n
        If you are interested in more parameter estimation, or have any other suggestions, please leave a message in the comment section.
        Comparison before and after reverse Function
        \n
    """)

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

st.success(
    """
    Let's take a look at the reverse tool and see the difference of the extracted information.
    """,
    icon="🗺",
)
option1 = st.selectbox(
    'Which Attack you would like to see if there is a difference made by the anti-attack function?',
    ('Cut',
     'Scale',
     'Cut + Scale',
     'Rotate',
     'Block',
     'Salt & Pepper Noise',
     'Adjust Brightness',
     'Warp',
     'Color Rotation',
     'Fourier Attack'))

st.write('You selected: ', option1)
if option1 == 'Cut' or option1 == 'Scale' or option1 == 'Cut + Scale':
    on = st.toggle('I will Provide Attack Parameters')
col1, col2 = st.columns(2)

if option1 == 'Cut':
    if on:
        coordinate = col1.text_input('Please provide your Attacking parameters: loc = ', ((0.1, 0.2), (0.5, 0.5)))
        coordinate = coordinate.replace(" ", "")
        coordinate = coordinate.replace("(", "")
        coordinate = coordinate.replace(")", "")
        x = coordinate.split(",")
        x1, y1, x2, y2 = int(w * float(x[0])), int(h * float(x[1])), int(w * float(x[2])), int(h * float(x[3]))
        password_1 = int(col2.text_input('Please input your password for extracting watermark:', '350'))
        Attacked = cv2.imread('Attacked_'+option1+'.jpg')
        img_recover = recover_crop(tem_img=Attacked,
                                   loc=(x1, y1, x2, y2), image_o_shape=ori_img_shape)
        bwm1 = WaterMark(password_wm=password_1, password_img=1)
        wm_extract_re = bwm1.extract(embed_img=img_recover, wm_shape=len_wm, mode='str')
        simi_score_re = fuzz.ratio(wm,wm_extract_re)
    else:
        # estimate crop attack parameters:
        Attacked = cv2.imread('Attacked_' + option1 + '.jpg')
        (x1, y1, x2, y2), image_o_shape, score, scale_infer = estimate_crop_parameters(ori_img=watermarked.astype(np.float32),
                                                                                       tem_img=Attacked.astype(np.float32),
                                                                                       scale=(1, 1), search_num=None)
        col2.write('The estimate parameters is :'+ str((x1/1000, y1/1000, x2/1000, y2/1000)))
        password_1 = int(col2.text_input('Please input your password for extracting watermark:', '350'))
        img_recover = recover_crop(tem_img=Attacked,
                                   loc=(x1, y1, x2, y2), image_o_shape=image_o_shape)
        bwm1 = WaterMark(password_wm=password_1, password_img=1)
        wm_extract_re = bwm1.extract(embed_img=img_recover, wm_shape=len_wm, mode='str')
        simi_score_re = fuzz.ratio(wm, wm_extract_re)
if option1 == 'Scale':
    if on:
        coordinate = col1.text_input('Please provide your Attacking parameters: shape = ', (400, 300))
    else:
        pass
    password_1 = int(col2.text_input('Please provide your password for extracting watermark:', password))
    Attacked = cv2.imread('Attacked_' + option1 + '.jpg')
    img_recover = att.resize_att(input_img=Attacked, out_shape=ori_img_shape[::-1])
    bwm1 = WaterMark(password_wm=password_1, password_img=1)
    wm_extract_re = bwm1.extract(embed_img=img_recover, wm_shape=len_wm, mode='str')
    simi_score_re = fuzz.ratio(wm,wm_extract_re)
if option1 == 'Cut + Scale':
    if on:
        coordinate = col1.text_input('Please provide your Attacking parameters: loc = ', ((0.1, 0.2), (0.5, 0.5)))
        scale = col1.text_input('scale = ', 0.6)
        coordinate = coordinate.replace(" ", "")
        coordinate = coordinate.replace("(", "")
        coordinate = coordinate.replace(")", "")
        x = coordinate.split(",")
        x1, y1, x2, y2 = int(w * float(x[0])), int(h * float(x[1])), int(w * float(x[2])), int(h * float(x[3]))
        password_1 = int(col2.text_input('Please input your password for extracting watermark:', '350'))
        Attacked = cv2.imread('Attacked_'+option1+'.jpg')
        img_recover = recover_crop(tem_img=Attacked, loc=(x1, y1, x2, y2), image_o_shape=ori_img_shape)
        bwm1 = WaterMark(password_wm=password_1, password_img=1)
        wm_extract_re = bwm1.extract(embed_img=img_recover, wm_shape=len_wm, mode='str')
        simi_score_re = fuzz.ratio(wm,wm_extract_re)
    else:
        # estimate crop attack parameters:
        Attacked = cv2.imread('Attacked_' + option1 + '.jpg')
        (x1, y1, x2, y2), image_o_shape, score, scale_infer = estimate_crop_parameters(ori_img=watermarked.astype(np.float32),
                                                                                       tem_img=Attacked.astype(np.float32),
                                                                                       scale=(0.5, 2), search_num=200)

        col2.write('The estimate parameters is : '+ str((x1/1000, y1/1000, x2/1000, y2/1000)) + ', Estimated Scale is : ' + str(1/scale_infer))
        password_1 = int(col2.text_input('Please input your password for extracting watermark:', '350'))
        img_recover = recover_crop(tem_img=Attacked,
                                   loc=(x1, y1, x2, y2), image_o_shape=image_o_shape)
        bwm1 = WaterMark(password_wm=password_1, password_img=1)
        wm_extract_re = bwm1.extract(embed_img=img_recover, wm_shape=len_wm, mode='str')
        simi_score_re = fuzz.ratio(wm, wm_extract_re)
if option1 == 'Rotate':
    angle = float(col1.text_input('Please provide your Attacking parameters: angle = ', 60))
    password_1 = int(col2.text_input('Please input your password for extracting watermark:', '350'))
    Attacked = cv2.imread('Attacked_'+option1+'.jpg')
    img_recover = att.rot_att(input_img=Attacked, angle=-angle)
    bwm1 = WaterMark(password_wm=password_1, password_img=1)
    wm_extract_re = bwm1.extract(embed_img=img_recover, wm_shape=len_wm, mode='str')
    simi_score_re = fuzz.ratio(wm,wm_extract_re)
if option1 == 'Fourier Attack':
    threshold2 = int(col1.text_input('Please provide your Attacking parameters: threshold = ', 25))
    password_1 = int(col2.text_input('Please input your password for extracting watermark:', '350'))
    # Attacked = cv2.imread('Attacked_'+option1+'.jpg')
    Attacked, filtered = att.fft_att2(image=watermarked, threshold=int(threshold))
    gray_recover=watermarked.astype(np.float32)[:,:,0]
    img_recover = att.anti_fft_att2(input_img=Attacked, color1 = filtered[:, :, 1], color2 = filtered[:, :, 2], threshold = threshold2)
    img_recover[:,:,0] = gray_recover
    bwm1 = WaterMark(password_wm=password_1, password_img=1)
    wm_extract_re = bwm1.extract(embed_img=img_recover, wm_shape=len_wm, mode='str')
    simi_score_re = fuzz.ratio(wm,wm_extract_re)


col1.write("Attacked Image")
col1.write("\n")

col1.image('Attacked_'+option1+'.jpg')
col1.write("The extracted Information is: "+ "\n"+ "***" + wm_extract + "***")
col1.write("The Similarity Score: " + "***" + str(simi_score) + "***")



if option1 == 'Block' or option1 =='Salt & Pepper Noise' or option1 == 'Adjust Brightness':
    col2.markdown("""
        The similarity score is pretty high, so there's no need to reverse this attack.
        \n
        You can try again with other attacks.
    """)
else:
    col2.write("Reversed Image")
    col2.write("\n")
    cv2.imwrite('Reverse.jpg', img_recover)
    col2.image('Reverse.jpg')
    col2.write("The extracted Information is: "+ "\n"+ "***" + wm_extract_re + "***")
    col2.write("The Similarity Score: "+ "***" + str(simi_score_re) + "***")

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

# st.subheader('Robustness Test - On large dataset')
# st.markdown("""
#     ######################### Still in Process ###########################
#     \n
#     Need to add the comments of why I test on a large dataset
#     \n
#     Show the Outcome
#     \n
#     In a table format
#     \n
#      ######################### Still in Process ###########################
# """)
