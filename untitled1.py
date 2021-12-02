# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:26:40 2020


"""
import tkinter
import PIL
import os
import PIL.Image, PIL.ImageTk
from age_emotion import fun,fun1
#from record import ui

import pyaudio
import wave
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import glob 



from sklearn.metrics import pairwise
import pickle
import tensorflow as tf
from tkinter import *
from tkinter import messagebox
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw
import PIL.ImageOps
from keras.models import load_model
import tkinter as tk
from PIL import ImageTk, Image

json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")
actualvalues={
'0':'Female Angry',
'1':'Female Calm',
'2':'Female Fearful',
'3':'Female Happy',
'4':'Female Sad',
'5':'Male Angry',
'6':'Male Calm',
'7':'Male Fearful',
'8':'Male Happy',
'9':'Male Sad'
}

class ui:
    def __init__(self):
        
        
        self.root = tk.Tk()
        self.frame = tk.Frame(self.root)
        self.frame.pack()
        #self.b= StringVar()
        
        self.button = tk.Button(self.frame,padx=16,pady=4,bd=4,fg="red",font=('arial',18,'bold'),width=24,height=3,bg="gray25", text="Recording",command=self.record)
        self.button.pack(side=tk.LEFT)


        self.slogan = tk.Button(self.frame,padx=16,pady=4,bd=4,fg="white",font=('arial',18,'bold'),width=24,height=3,bg="gray25",text="Predict",command=self.test)

        self.slogan.pack(side=tk.LEFT)
        
        
        #self.w = tk.Label(self.root, textvariable=self.b ,fg="blue",font=('comic sans ms',22,'bold'),width=24,height=3)
        #self.w.pack()


        

        self.root.mainloop()

    def record(self):

        CHUNK = 1024 
        FORMAT = pyaudio.paInt16 #paInt8
        CHANNELS = 2 
        RATE = 44100 #sample rate
        RECORD_SECONDS = 4
        WAVE_OUTPUT_FILENAME = "output10.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK) #buffer

        print("* Recording....")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data) # 2 bytes(16 bits) per channel

        print("* Recording Done")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    def test(self):
    
        X, sample_rate = librosa.load('output10.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
        featurelive = mfccs
        livedf2 = featurelive
        livedf2= pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T
        twodim= np.expand_dims(livedf2, axis=2)
        livepreds = loaded_model.predict(twodim, 
                                 batch_size=32, 
                                 verbose=1)

        livepreds1=livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()

        lb = LabelEncoder()
        liveabc=lb.fit_transform(liveabc)
        livepredictions = (lb.inverse_transform((liveabc)))
        livepredictions=str(list(livepredictions))[1:-1]
        ans=livepredictions
        for key,value in actualvalues.items():
            if(ans==key):
                break
        #self.b.set(value)
        
        root1 = tk.Tk()
        frame = tk.Frame(root1)
        frame.pack()
        
        w = tk.Label(root1, text=value ,fg="blue",font=('comic sans ms',22,'bold'),width=24,height=3)
        w.pack()

        self.root.mainloop()

            


        




def main():
    def letrecogmain():
        pass
    def gesturemain():
        window.destroy()
        gesture()
    window=tkinter.Tk()
    _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
    _fgcolor = '#000000'  # X11 color: 'black'
    _compcolor = '#ffffff' # X11 color: 'white'
    _ana1color = '#ffffff' # X11 color: 'white'
    _ana2color = '#ffffff' # X11 color: 'white'

    #face="C:/Users/Gurkirat/Desktop/face.jpg"
    #img = ImageTk.PhotoImage(Image.open(face))
    #voice="C:/Users/Gurkirat/Desktop/voice.jpg"
    #img = ImageTk.PhotoImage(Image.open(voice))
    
    window.title('Emotion, Age and Gender Detection')
    window.geometry('1000x700')
    window = Frame( window)
    window.place(relx=0.02, rely=0.03, relheight=0.94, relwidth=0.96)
    window.configure(relief=GROOVE)
    window.configure(borderwidth="2")
    window.configure(relief=GROOVE)
    window.configure(background="#d9d9d9")
    window.configure(highlightbackground="#d9d9d9")
    window.configure(highlightcolor="black")
    window.configure(width=925)



    
    window.configure(background="#d9d9d9")
    window.configure(highlightbackground="#d9d9d9")
    window.configure(highlightcolor="black")
    lblHeading = Label(window,text = "\t\t\t Emotion, Age and Gender Detection \t\t\t\t",font=('Calibri',26,'bold'), bg="#d9d9d9",height=3).pack()
    b0=Button(window,padx=16,pady=4,bd=4,fg="white",font=('arial',16,'bold'),width=21,height=2,bg="gray25",text='Image Recognition',command=fun1)
    b0.place(x=70,y=150)
    b1=Button(window,padx=16,pady=4,bd=4,fg="white",font=('arial',16,'bold'),width=21,height=2,bg="gray25",text='Face Recognition',command=fun)
    b1.place(x=70,y=327)
    b2=Button(window,padx=16,pady=4,bd=4,fg="white",font=('arial',16,'bold'),width=21,height=2,bg="gray25",text='Voice Recognition',command=ui)
    b2.place(x=70,y=500)
    
    #facial="C:/Users/Gurkirat/Desktop/istockfae.jpg"
    #img0 = ImageTk.PhotoImage(Image.open(facial))
    #panel = Button(window, image = img0, command=fun1)
    #panel.image = img0
    #panel.place(x=600,y=150)
    #face="C:/Users/Gurkirat/Desktop/face.jpg"
    #img1 = ImageTk.PhotoImage(Image.open(face))
    #panel = Button(window, image = img1,command=fun)
    #panel.image = img1
    #panel.place(x=600,y=327)
    #voice="C:/Users/Gurkirat/Desktop/voice.jpg"
    #img2 = ImageTk.PhotoImage(Image.open(voice))
    #panel2 = Button(window, image = img2,command=ui)
    #panel2.image = img2
    #panel2.place(x=600,y=500)
    #window.mainloop()



if __name__=='__main__':
    main()
    
