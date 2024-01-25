# importing libraries
import streamlit as st
import streamlit_lottie as st_lottie
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import requests
import librosa
import librosa.display

# tensorflow
from tensorflow.keras.models import load_model

# set page config

st.set_page_config(page_title='Sonic sentiment', page_icon=':rocket:', layout='wide')

# model

model = load_model('audio_conv1d.h5')

# loading animations

def loader_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def model_inference(audio_file):
    y, sr = librosa.load(audio_file)
    D = librosa.stft(y)
    spectrogram = librosa.amplitude_to_db(abs(D))
    spectrogram_resized = cv2.resize(spectrogram, (40, 1025))
    #img_array = np.expand_dims(spectrogram_resized, axis=0)
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    predictions = model.predict(np.expand_dims(spectrogram_resized, axis=0))
    predicted_class = np.argmax(predictions[0], axis=-1)
    label_encoder_mapping = {0: 'Laryngozele', 1: 'Normal', 2: 'Vox senilis'}
    plt.savefig('librosa.jpg')
    plt.close()
    st.write(label_encoder_mapping[predicted_class])
    st.image('librosa.jpg')
    
# loading animations

music = loader_url('https://lottie.host/1ee5b4b7-7267-44d7-8ce2-3aa911aba9c7/mFSHfTQGl8.json')  

# Home page sidebar

st.markdown("<h1 style='text-align: center;'>Sonic sentiment</h1>", unsafe_allow_html=True)

with st.container():
    l,m = st.columns((1,2))
    with l:
        st.lottie(music, height=300, key='music')
    with m:
        st.title('##')
        #st.markdown("<h3 style='text-align: center; '>Upload the file</h3>", unsafe_allow_html=True)
        audio = st.file_uploader('Upload the audio: ', type=['wav', 'mp3'])
        if st.button('Enter') and audio is not None:
            model_inference(audio)
            os.remove('librosa.jpg')
        

        