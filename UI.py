
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import os
import cv2
import sys
import numpy as np
from tensorflow.keras.models import load_model
from skimage import transform
from PIL import Image, ImageTk
import numpy
import requests
import json

model = load_model('model/vgg19_transfer_200ep_bestsave.h5')
result_dict = {0 :"NG", 1 : "OK"}

# croping non-interested area using statistical operations over interest points 
def non_interest_point_croping(img):
    orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    
    x = np.array([keypoint.pt[0] for keypoint in kp]).astype(np.int16)
    y = np.array([keypoint.pt[1] for keypoint in kp]).astype(np.int16)

    xstd = np.std(x)
    ystd = np.std(y)
    xmean = np.mean(x)
    ymean = np.mean(y)
    # print (xstd,ystd,xmean,ymean)
    x0,y0 = int(xmean - 2*xstd), int(ymean - 2*ystd)
    x1,y1 = int(xmean + 2*xstd), int(ymean + 2*ystd)

    if x0<min(x):
        x0=int(min(x))
    if y0<min(y):
        y0=int(min(y))

    if x1> max(x):
        x1 =int(max(x))
    if y1> max(y):
        y1 = int(max(y))
    
    return img[x0:x1,y0:y1,:]

def load(np_image):
    # np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    img = np_image[0:400,20:500,:]
    img = non_interest_point_croping(img)
    img = cv2.resize(img,(50,50),interpolation=cv2.INTER_AREA)
    ret,img = cv2.threshold(img,150,255,cv2.THRESH_TOZERO)
    x = np.expand_dims(img  , axis=0)
    return x


def change_labelcolor(count, category):
    global label1, label2, label3, label4, OK, NG, num_ok, num_ng
    
#     print (f"response is {num_ok} and {type(num_ng)}")
    list_labels = [label1, label2, label3, label4]
    if category == "OK":
        list_labels[count].config(bg="green", text = f"Cross Oil Hole {count+1} is OK")
        OK += 1
        num_ok.config(text = f"{OK}")
        
    elif category == "NG":
        list_labels[count].config(bg="red", text = f"Cross Oil Hole {count+1} is NG")
        NG += 1
        num_ng.config(text = f"{NG}")




def prompt_ok(event = 0):
    
    global cancel, button, button1, button2
    cancel = True

    button.place_forget()
    button1 = tk.Button(canvas2, text="Process", command=saveAndExit)
    button2 = tk.Button(canvas2, text="Capture Again", command=resume)
    button1.place(anchor=tk.CENTER, relx=0.2, rely=0.9, width=150, height=50)
    button2.place(anchor=tk.CENTER, relx=0.8, rely=0.9, width=150, height=50)
    button1.focus()
    
def reset():
    global id_var, label1, label2, label3, label4, count, num_ng, num_ok
    list_labels = [label1, label2, label3, label4]
    id_var.set("")
    count = 4
    num_ng.config(text = "0")
    num_ok.config(text = "0")
    for i in range(4):
        list_labels[i].config(text=f'       Cross Oil Hole {i+1}        ', fg='black', bg='white',  font=("Helvetica", 16))



def saveAndExit(event = 0):
    global prevImg, count, id_num, learn
    
    try:
        if (count > 0) and (id_num.get() != ""):
            ### Image reading and fetching data
            img = load(prevImg)
            print (f"******************{type(prevImg), type(img)}*********************")
            prediction = model.predict_classes(img)[0]
            print (f"The prediction is {prediction}")
            response = result_dict[prediction[0]]
            messagebox.showinfo("Response",f"The Image is {response}")
            
            ###changing the label color
            change_labelcolor(4-count, response) ###using 4 - since count is 4 and list index should be 0
            count -= 1
            
        elif id_num.get() == "":
            messagebox.showwarning("Missing Info", "Enter Part Number to continue")
        else:
            messagebox.showwarning("Cant Process more", "You have taken 4 photos for this engine. Click Inpection Now")
    
    except Exception as e:
        messagebox.showinfo("Error",e)
    
    
    #####saving the images
    folder_name = f"results/{id_num.get()}_OK_{datetime.today().strftime('%m%d%Y')}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    prevImg = numpy.array(prevImg, dtype="uint8")
    prevImg = prevImg.astype(numpy.uint8)
    prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{folder_name}/img_{4-count}.jpg", prevImg)



#####Done
def resume(event = 0):
    global button1, button2, button, lmain, cancel

    cancel = False

    button1.place_forget()
    button2.place_forget()

    mainWindow.bind('<Return>', prompt_ok)
    button.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=300, height=50)
    lmain.after(10, show_frame)
    

#######DOne
def changeCam(event=0, nextCam=-1):
    global camIndex, cap, fileName

    if nextCam == -1:
        camIndex += 1
    else:
        camIndex = nextCam
    del(cap)
    cap = cv2.VideoCapture(camIndex)

    #try to get a frame, if it returns nothing
    success, frame = cap.read()
    if not success:
        camIndex = 0
        del(cap)
        cap = cv2.VideoCapture(camIndex)

    f = open(fileName, 'w')
    f.write(str(camIndex))
    f.close()               

#####DOne               
def show_frame():
    global cancel, prevImg, button

    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    prevImg = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=prevImg)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    if not cancel:
        lmain.after(10, show_frame)
        
def on_closing():
    global cap
    cap.release()
    cv2.destroyAllWindows()
    mainWindow.destroy()



if __name__ == "__main__":
    
    fileName = os.environ['ALLUSERSPROFILE'] + "\WebcamCap.txt"
    cancel = False
    
    try:
        f = open(fileName, 'r')
        camIndex = int(f.readline())
    except:
        camIndex = 0

    cap = cv2.VideoCapture(camIndex)
    capWidth = cap.get(3)
    capHeight = cap.get(4)

    success, frame = cap.read()
    if not success:
        if camIndex == 0:
            print("Error, No webcam found!")
            sys.exit(1)
        else:
            changeCam(nextCam=0)
            success, frame = cap.read()
            if not success:
                print("Error, No webcam found!")
                sys.exit(1)
                
    ####variable
    count = 4
    OK = 0
    NG = 0
    
    ####main window title and inspection
    mainWindow = tk.Tk(screenName="Camera Capture")
    mainWindow.title("Auto Engine Inspection")
    width, height = 900,600
    mainWindow.geometry(f"{width}x{height}")
#     mainWindow.configure(background="#f8f8f8")
    mainWindow.resizable(width=True, height=True)
    mainWindow.bind('<Escape>', lambda e: mainWindow.quit()) ##binding operation here
    
    canvas1 = tk.Canvas(mainWindow )
    canvas2 = tk.Canvas(mainWindow )
    canvas3 = tk.Canvas(mainWindow )
    canvas4 = tk.Canvas(mainWindow )
    canvas5 = tk.Canvas(mainWindow )
    canvas6 = tk.Canvas(mainWindow )
    
    canvas1.grid(row=0, column = 0, padx = 5, pady = 5)
    canvas2.grid(row=0, column = 1, padx = 5, pady = 5)
    canvas3.grid(row=0, column = 2, padx = 5, pady = 5)
    canvas4.grid(row=1, column = 0, padx = 5, pady = 5)
    canvas5.grid(row=1, column = 1, padx = 5, pady = 5)
    canvas6.grid(row=1, column = 2, padx = 5, pady = 5)
    
    ##############################################################################################Canvas 1  
    heading = tk.Label(canvas1, text = "Borescope Inspection Auto  Judgement",  font=("Helvetica", 20))
    heading.pack(side=tk.TOP, anchor=tk.NW)
    
    #--------Model
    label_model = tk.Label(canvas1, text= "           Model             ",bg ="#0066ff" ,fg = "white",font=("Helvetica", 16),relief=tk.RAISED)
    label_model.pack(pady=10)
    
    id_model = tk.StringVar()
    id_model_input = tk.Entry(canvas1, textvariable = id_model, font=('calibre',26,'normal'))
    id_model_input.pack()
    #--------Part Number
    label = tk.Label(canvas1, text= "           Part Number Input             ",bg ="#0066ff" ,fg = "white",font=("Helvetica", 16),relief=tk.RAISED)
    label.pack(pady = 10)
    
    id_var = tk.StringVar()
    id_num = tk.Entry(canvas1, textvariable = id_var, font=('calibre',26,'normal'))
    id_num.pack(padx = 50)
    ############################################################################################### canvas 2
    lmain = tk.Label(canvas2, compound=tk.CENTER, anchor=tk.CENTER, relief=tk.RAISED)
    button = tk.Button(canvas2, text="Capture", font=("Helvetica", 10),command=prompt_ok)
    button_changeCam = tk.Button(canvas2, text="Switch Camera", font=("Helvetica", 10),command=changeCam)

    lmain.pack()
    id_num.pack( side = tk.LEFT )
    button.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=height//2, height=50)
    button.focus()
    button_changeCam.place(bordermode=tk.INSIDE, relx=0.85, rely=0.1, anchor=tk.CENTER, width=150, height=50)
    
    ################################################################################################## canvas 4
    label_c4 = tk.Label(canvas4, text= "       Inspection Oil Hole           ",bg ="#0066ff" ,fg = "white",font=("Helvetica", 16),relief=tk.RAISED)
    label_c4.pack(padx = 10,pady= 10)
    label1 = tk.Label(canvas4, text='       Cross Oil Hole 1            ', fg='black', bg='#92d050',  font=("calibre", 16), relief=tk.RAISED)
    label2 = tk.Label(canvas4, text='       Cross Oil Hole 2           ', fg='black', bg='#92d050', font=("calibre", 16), relief=tk.RAISED)
    label3 = tk.Label(canvas4, text='       Cross Oil Hole 3           ', fg='black', bg='#92d050', font=("calibre", 16), relief=tk.RAISED)
    label4 = tk.Label(canvas4, text='       Cross Oil Hole 4           ', fg='black', bg='#92d050', font=("calibre", 16), relief=tk.RAISED)
    label1.pack(padx=6, pady=6)
    label2.pack(padx=6, pady=6)
    label3.pack(padx=6, pady=6)
    label4.pack(padx=6, pady=6)
    
    #--------------------------------------------------------------- canvas5
    
    # Put the image into a canvas compatible class, and stick in an
    # arbitrary variable to the garbage collector doesn't destroy it

    img1= Image.open("images/nk_ng_sample.png")
    img1 = img1.resize((646,300))
    img1 = ImageTk.PhotoImage(img1)
    panel = tk.Label(canvas5, image = img1)
    panel.pack(side = "bottom", fill = "both", expand = "yes")
    
    #------------------------------------------------------------------canvas6
    inspection = tk.Button(canvas6, text="INSPECTION COMPLETE",bg="#0066ff", fg = "white",font=("Calibre", 18),command = reset)
    inspection.place(bordermode=tk.INSIDE, relx=0.5, rely=0.5, anchor=tk.CENTER, width=height//2, height=50)
    inspection.focus()
    show_frame()
    
    #----------------------------------------------------------------- canvas 3
    inspection_result = tk.Label(canvas3, text='       Inspection Result           ', bg ="#0066ff" ,fg = "white",font=("Helvetica", 16),relief=tk.RAISED)
    inspection_result.grid(row=0, columnspan = 2, sticky = tk.N)
    #-------count_OK
    count_ok = tk.Label(canvas3, text="OK Qty",fg='black', bg='#92d050',font=("Helvetica", 12))
    count_ok.grid(row=1,column=0, padx = 5, pady=10)
    #-------count_NG
    count_ng = tk.Label(canvas3, text="NG Qty",fg='black', bg='#ff0000',font=("Helvetica", 12))
    count_ng.grid(row=2,column=0, padx = 5, pady=10)
    num_ok = tk.Label(canvas3, text="0",font=("Helvetica", 12))
    num_ok.grid(row=1,column=1)
    num_ng = tk.Label(canvas3, text="0",font=("Helvetica", 12))
    num_ng.grid(row=2,column=1)

    
    mainWindow.protocol("WM_DELETE_WINDOW", on_closing)
    mainWindow.mainloop()





