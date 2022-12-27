import os
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile

import sounddevice as sd
import soundfile as sf
from tkinter import *


def Voice_rec():
    fs = 48000

    # seconds
    duration = 3
    myrecording = sd.rec(int(duration * fs),
                         samplerate=fs, channels=2)
    sd.wait()
    filename = str(file_text_box.get(1.0, END))
    filename = filename.replace('\n', ' ')
    os.chdir('data_from_recordings')
    print(os.getcwd())
    return sf.write(filename + ".wav", myrecording, fs)


def save_text():
    name_of_file = file_text_box.get(1.0, END)
    name_of_file = name_of_file.replace('\n', ' ')
    text_file = open(name_of_file + ".txt", "w")
    text_file.write(my_text_box.get(1.0, END))
    text_file.close()

# Create an instance of tkinter window with a frame
win = Tk()
win.title("Txt and WAV generator for easy data recording and labeling")
win.geometry("500x250")
fm = Frame(win)

# Creating a label
label_1 = Label(fm, text="The textbox below is the name of the file\n It will be the same for the recording and the "
                         "text\n example : tristan_oui_3")
label_1.pack(side=TOP, expand=YES)

# Create a text box to input file name
file_text_box = Text(fm, height=1, width=30)
file_text_box.pack(side=TOP, expand=YES)

# Creating a label
label_2 = Label(fm, text="Here input the text you want to speak \n then press the save text file button\nexample : oui")
label_2.pack(side=TOP, expand=YES)

# Creating a text box widget with label
my_text_box = Text(fm, height=1, width=30)
my_text_box.pack(side=TOP, expand=YES)

# Create a button to save the text
save = Button(fm, text="Save text file", command=save_text)
save.pack(side=TOP, expand=YES)

# Creating a label
label_2 = Label(fm, text="Click the button below when you want to record \nIt will record for 5s and be saved "
                         "automatically after with the same name file")
label_2.pack(side=TOP, expand=YES)

# Create a button to start the recording and save it in wav
voice_recorder = Button(fm, text="Start and save voice recording", command=Voice_rec)
voice_recorder.pack(side=TOP, expand=YES)
fm.pack(fill=BOTH, expand=YES)

win.mainloop()
