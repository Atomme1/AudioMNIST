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

    # Save as FLAC file at correct sampling rate
    # return sf.write('data_from_recordings/my_Audio_file.wav', myrecording, fs)
    filename = str(file_text_box.get(1.0, END))
    filename = filename.replace('\n', ' ')
    os.chdir('data_from_recordings')
    print(os.getcwd())
    return sf.write(filename + ".wav", myrecording, fs)


def save_text():
    file = filedialog.asksaveasfile(initialdir=os.getcwd(),
                                    defaultextension='.txt',
                                    filetypes=[
                                        ("Text file", ".txt"),
                                        ("All files", ".*"),
                                    ])
    if file is None:
        return
    filetext = str(my_text_box.get(1.0, END))
    file.write(filetext)
    file.close()


# Create an instance of tkinter window
win = Tk()
win.geometry("500x500")

label = Label(win, text="The text below is the name of the file\n that will be the same for the recording and audio")
label.grid(column=0, row=0)

# Create a text box to input file name
file_text_box = Text(win, height=1, width=10)
file_text_box.grid(column=0, row=1)

# Creating a text box widget with label
label = Label(win, text="Here input the text you want to speak \n then press the save button")
label.grid(column=0, row=2)
my_text_box = Text(win, height=1, width=15)
my_text_box.grid(column=0, row=3)

# Create a button to save the text
save = Button(win, text="Save File", command=save_text)
save.grid(column=0, row=4)

# Create a button to start the recording and save it in wav
voice_recorder = Button(win, text="Start voice recording", command=Voice_rec)
voice_recorder.grid(column=0, row=5)

win.mainloop()
