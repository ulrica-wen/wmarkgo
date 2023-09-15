import streamlit as st
from PIL import Image
import os
dir_path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(dir_path)

st.set_page_config(
    page_title="DeepFloyd Image Generation",
    page_icon=":cityscape:",
    layout='wide'
)

st.title('🏙️ DeepFloyd Image Generation')

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')
st.subheader('Generation')
st.markdown(
    """
    The following is an example of input prompts:
    """
)
col1, col2, col3, col4 = st.columns([4,2,3,3])
with col2:
    genre = st.radio(' ', ['A photo of a', 'An oil painting of a'])
    character = st.radio(' ', ['fuzzy panda', 'dog'])
with col3:
    outfit = st.radio(' ', ['wearing a cowboy hat and red shirt', 'wearing a sunglasses and black leather jacket'])
    # st.radio(' ', ['red shirt', 'black leather jacket'])
with col4:
    activity = st.radio(' ', ['playing a guitar', 'riding a bike'])
    place = st.radio(' ', ['in the New York Street.', 'on a beach.'])

col1.image(Image.open(dir_path+'/image outcome/' + genre+'/'+character+'/'
                    +outfit +'/' + activity+'/'+place+'png'))
st.markdown('\n')
st.markdown('\n')
st.markdown('\n')
st.subheader('Generation - With Your Own Prompt')
st.markdown(
    """
    ############This section is still in progress############
    \n
    You can try with your own prompt to see the image generated with DeepFloyd:
    """
)
title = st.text_input('Prompt: ', 'A photo of a fuzzy panda wearing a sunglasses and black leather jacket riding a bike in the New York Street.')
st.image(Image.open(dir_path+'/image outcome/A photo of a/fuzzy panda/wearing a sunglasses and black leather jacket/riding a bike/in the New York Street.png'))

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

st.subheader('Style Transfer')
st.markdown(
    """
    The following is an example of Style Transfer:
    """
)

col1, col2, col3 = st.columns([3,2,3])

col1.image(Image.open(dir_path+'/image outcome/style transfer/ori.jpg'))
genre = col2.radio(
    "What's your favorite style",
    [":rainbow[Lego]", "***Zombie***", "Origami","Anime :movie_camera:"],
    captions = ["####adding comments####.", "####adding comments####.", "####adding comments####.", "####adding comments####."])
if genre == ":rainbow[Lego]":
    col3.image(Image.open(dir_path+'/image outcome/style transfer/lego.jpg'))
elif genre == "***Zombie***":
    col3.image(Image.open(dir_path+'/image outcome/style transfer/zombie.jpg'))
elif genre == "Origami":
    col3.image(Image.open(dir_path+'/image outcome/style transfer/origami.jpg'))
else:
    col3.image(Image.open(dir_path+'/image outcome/style transfer/anime.jpg'))


st.markdown('\n')
st.markdown('\n')
st.markdown('\n')


st.info(
    """
    ############This section is still in progress############
    \n
    Try it with your own style prompt! You can enter description like:\n 
    Anime style in 1990, \n
    An oil painting by Van gogh\n 
    and etc.
    """,
    icon="👻",
)

col1, col2= st.columns(2)
style = col1.text_input('Style: ', 'Anime style in 1990')
col2.image(Image.open(dir_path+'/image outcome/style transfer/anime.jpg'))