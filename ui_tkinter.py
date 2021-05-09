
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tkinter import *
import cv2
from PIL import Image, ImageTk
from threading import Thread
import model
import math
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import datetime as dt
import time


tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def layout_():
    global canvas, button_draw, label_car1, button_start, frame4, frame3
    frame_left = Frame(window, bg="#19e0d6", width=220,height=850)
    frame_left.grid(row=0, column=0, pady=5)
    btn_dashboard = Button(frame_left, text="Dashboard")
    btn_dashboard.pack( padx=10, pady=10)
    btn_Model = Button(frame_left, text="Model")
    btn_Model.pack( padx=10, pady=10)
    btn_Stream = Button(frame_left, text="Stream")
    btn_Stream.pack( padx=10, pady=10)
    btn_Setting = Button(frame_left, text="Setting")
    btn_Setting.pack( padx=10, pady=10)
    frame_left.propagate(0)

    frame_right = Frame(window, bg="#dedbd7", width=1300, height=850)
    frame_right.grid(row=0, column=1, pady=5)
    frame_right.propagate(0)


    frame0 = Frame(frame_right, bg="blue", width=220, height=500)
    frame0.grid(row=0, column=2, padx=5, pady=5)
    frame0.propagate(0)

    frame1 = LabelFrame(frame0, bg="pink")
    frame1.pack(pady=20)
    label_car = Label(frame1, text="Car", bg="pink", font=("Arial", 20))
    label_car.grid(row=0, column=0, sticky=W)
    label_motor = Label(frame1, text="Motor", bg = "pink",font=("Arial",20))
    label_motor.grid(row=1, column=0, sticky=W)
    label_bus = Label(frame1, text="Bus", bg="pink", font=("Arial", 20))
    label_bus.grid(row=2,column=0,sticky=W)
    label_track = Label(frame1, text="Track", bg = "pink",font=("Arial",20))
    label_track.grid(row=3,column=0,sticky=W)
    label_person = Label(frame1, text="Person", bg = "pink",font=("Arial",20))
    label_person.grid(row=4,column=0,sticky=W)

    label_car1 = Label(frame1, text="0", bg = "pink",font=("Arial",20))
    label_car1.grid(row=0,column=1,sticky=W,padx=10)
    label_motor1 = Label(frame1, text="0", bg = "pink",font=("Arial",20))
    label_motor1.grid(row=1,column=1,sticky=W,padx=10)
    label_bus1 = Label(frame1, text="0", bg = "pink",font=("Arial",20))
    label_bus1.grid(row=2,column=1,sticky=W,padx=10)
    label_track1 = Label(frame1, text="0", bg = "pink",font=("Arial",20))
    label_track1.grid(row=3,column=1,sticky=W,padx=10)
    label_person1 = Label(frame1, text="0", bg = "pink",font=("Arial",20))
    label_person1.grid(row=4,column=1,sticky=W,padx=10)

    frame2 = LabelFrame(frame0, bg = "pink")
    frame2.pack(pady=20)
    label = Label(frame2, text="Car", bg = "pink",font=("Arial",20))
    label.grid(row=0,column=0,sticky=W)
    label = Label(frame2, text="Motor", bg = "pink",font=("Arial",20))
    label.grid(row=1,column=0,sticky=W)
    label = Label(frame2, text="Bus", bg = "pink",font=("Arial",20))
    label.grid(row=2,column=0,sticky=W)
    label = Label(frame2, text="Track", bg = "pink",font=("Arial",20))
    label.grid(row=3,column=0,sticky=W)
    label = Label(frame2, text="Person", bg = "pink",font=("Arial",20))
    label.grid(row=4,column=0,sticky=W)

    label = Label(frame2, text="0", bg="pink", font=("Arial", 20))
    label.grid(row=0, column=1, sticky=W, padx=10)
    label = Label(frame2, text="0", bg="pink", font=("Arial", 20))
    label.grid(row=1, column=1, sticky=W, padx=10)
    label = Label(frame2, text="0", bg="pink", font=("Arial", 20))
    label.grid(row=2, column=1, sticky=W, padx=10)
    label = Label(frame2, text="0", bg="pink", font=("Arial", 20))
    label.grid(row=3, column=1, sticky=W, padx=10)
    label = Label(frame2, text="0", bg = "pink", font=("Arial", 20))
    label.grid(row=4, column=1, sticky=W, padx=10)

    canvas = Canvas(frame_right, width=690, height=490, bd=5, bg='#3ffc00')
    canvas.grid(row=0, column=1, padx=15, pady=5)

    frame3 = Frame(frame_right, bg='yellow', width=220, height=500)
    frame3.grid(row=0, column=3, padx=5, pady=5)
    frame3.propagate(0)
    button_draw = Button(frame3, text="Draw Line", command=draw_line)
    button_draw.pack(padx=15, pady=15)
    button_rmline = Button(frame3, text="Remove line", command=removeline)
    button_rmline.pack(padx=15, pady=15)

    button = Button(frame3, text="Submit", command=window.quit)
    button.pack(padx=15, pady=15)

    button_slpoint = Button(frame3, text= "Select point", command=change_state)
    button_slpoint.pack(padx=15, pady=15)

    button_start = Button(frame3, text="Start",command=start_dectect)
    button_start.pack(padx=15,pady=15)

    frame4 = Frame(frame_right, bg='orange', width=1170, height=340)
    frame4.grid(row=1, column=1, columnspan=3, padx=15)
    frame4.propagate(0)


def set_coords(event):
    global x,y
    x = event.x
    y = event.y
    X.append(x)
    Y.append(y)

def removeline():
    X.clear()
    Y.clear()

def update_frame():
    global img1,img,canvas, curr_trackers,car_number,obj_cnt,frame_count
    ret, frame = video.read()
    img1 = cv2.resize(frame, (700, 500))
    if btn_draw == True:
        canvas.bind('<Button-1>', set_coords)
        n = len(X)
        if n >= 1:
            cv2.circle(img1, (X[0], Y[0]), 5, (0, 255, 255), thickness=-1)
            for i in range(1, n):
                cv2.line(img1, (X[i], Y[i]), (X[i - 1], Y[i - 1]), laser_line_color, 2)
                cv2.circle(img1, (X[i], Y[i]), 5, (0, 255, 255), thickness=-1)
    else:
        canvas.unbind('<Button-1>')
        n = len(X)
        if n >= 1:
            cv2.circle(img1, (X[0], Y[0]), 5, (0, 255, 255), thickness=-1)
            for i in range(1, n):
                cv2.line(img1, (X[i], Y[i]), (X[i - 1], Y[i - 1]), laser_line_color, 2)
                cv2.circle(img1, (X[i], Y[i]), 5, (0, 255, 255), thickness=-1)
            cv2.line(img1, (X[0], Y[0]), (X[-1],Y[-1]), laser_line_color, 2)
    if check_loadmd == True:
        tracking_detect(img1)
    label_car1.configure(text=len(curr_trackers))
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(image=Image.fromarray(img))
    canvas.create_image(0, 0, image=img, anchor=NW)
    window.after(15,update_frame)

def draw_line():
    global btn_draw
    if btn_draw == False:
        btn_draw = True
        button_draw.configure(text="Complete Draw")
    else:
        btn_draw =False
        button_draw.configure(text="Draw Line")
def get_box_info(box):
    (x, y, w, h) = [int(v) for v in box]
    center_X = int((x + (x + w)) / 2.0)
    center_Y = int((y + (y + h)) / 2.0)
    return x, y, w, h, center_X, center_Y
def change_state():
    global continuePlotting
    if continuePlotting == True:
        continuePlotting = False
    else:
        continuePlotting = True
        draw_chart()
def draw_chart():
    global canvas_chart
    f = Figure(figsize=(12,6))
    a = f.add_subplot(111)
    f.autofmt_xdate(rotation=45)
    a.set_ylabel("Total vehicle")
    a.grid()
    xs = []
    ys = []
    a.plot(xs,ys)
    canvas_chart = FigureCanvasTkAgg(f, master=frame4)
    canvas_chart.get_tk_widget().pack(side=LEFT)

    def animate(xs, ys):
        while True:
            xs.append(dt.datetime.now().strftime('%H:%M:%S'))
            ys.append(len(curr_trackers))
            #gioi han list
            xs = xs[-20:]
            ys = ys[-20:]

            a.clear()
            a.grid()
            a.plot(xs, ys)
            f.autofmt_xdate(rotation=45)

            canvas_chart.draw_idle()
            print(xs)
            print(ys)
            time.sleep(1)

    animate(xs,ys)
    #animation.FuncAnimation(f, animate, interval=100)

    '''toolbar = NavigationToolbar2Tk(canvas_chart, frame3)
    toolbar.update()
    canvas_chart._tkcanvas.pack()

    canvas_chart1 = FigureCanvasTkAgg(f, master=frame4)
    canvas_chart1.draw()
    canvas_chart1.get_tk_widget().pack(side= LEFT)'''

def tracking_detect(frame):
    global curr_trackers,car_number,obj_cnt,frame_count
    boxes = []
    imH = 500
    imW = 700
    x_point_1 = int(imW / 2)
    y_point_1 = imH
    x_point_2 = imW
    y_point_2 = int(imH / 2)
    distance_1_2 = math.sqrt((x_point_2 - x_point_1) ** 2 + (y_point_2 - y_point_1) ** 2)

    laser_line = imH - 100
    laser_line_color = (0, 0, 255)
    old_trackers = curr_trackers
    curr_trackers = []

    # duyệt qua các tracker cũ
    for car in old_trackers:
        tracker = car['tracker']
        (_, box) = tracker.update(frame)
        boxes.append(box)

        new_obj = dict()
        new_obj['tracker_id'] = car['tracker_id']
        new_obj['tracker'] = tracker

        # tính toán tâm đối tượng
        x, y, w, h, center_X, center_Y = model.get_box_info(box)

        # Ve hinh chu nhat quanh doi tuong
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Ve hinh tron tai tam doi tuong
        cv2.circle(frame, (center_X, center_Y), 4, (0, 255, 0), -1)


        if area_to_point(center_X,center_Y)!=area():
            # Neu vuot qua thi khong track nua ma dem xe
            laser_line_color = (0, 255, 255)
            car_number += 1

        else:
            # Con khong thi track tiep
            curr_trackers.append(new_obj)

    # Thuc hien object detection moi 5 frame
    if frame_count % 5 == 0:
        # Detect doi tuong
        boxes_d, classed = model.get_object(frame, detect_fn)

        for box in boxes_d:
            old_obj = False

            xd, yd, wd, hd, center_Xd, center_Yd = get_box_info(box)
            if st == 1:
                if area_to_point(center_Xd,center_Yd)==area():

                # Duyet qua cac box, neu sai lech giua doi tuong detect voi doi tuong da track ko qua max_distance thi coi nhu 1 doi tuong
                    if not model.is_old(center_Xd, center_Yd, boxes):
                        cv2.rectangle(frame, (xd, yd), ((xd + wd), (yd + hd)), (0, 255, 255), 2)
                        # Tao doi tuong tracker moi

                        tracker = cv2.TrackerMOSSE_create()

                        obj_cnt += 1
                        new_obj = dict()
                        tracker.init(frame, tuple(box))

                        new_obj['tracker_id'] = obj_cnt
                        new_obj['tracker'] = tracker

                        curr_trackers.append(new_obj)

    # Tang frame
    frame_count += 1
    return frame

def area():
    n=len(X)
    S=0
    if n>=3:
        for i in range(2,n):
            S += (1/2)*abs((X[i-1]-X[0])*(Y[i]-Y[0])-(X[i]-X[0])*(Y[i-1]-Y[0]))
    return S
def area_to_point(a,b):
    n=len(X)
    S=0
    for i in range(n-1):
        S += (1/2)*abs((X[i]-a)*(Y[i+1]-b)-(X[i+1]-a)*(Y[i]-b))
    S += (1/2)*abs((X[n-1]-a)*(Y[0]-b)-(X[0]-a)*(Y[n-1]-b))
    return S

def start_dectect():
    global st
    st= 1
    thread2 = Thread(target=draw_chart)
    thread2.start()
def load_model():
    global detect_fn, check_loadmd
    detect_fn = model.load_model()
    check_loadmd = True
    print("Load model complete")
if __name__ == "__main__":
    window = Tk()
    window.title("Traffic App")
    window.attributes("-fullscreen", True)
    #window.geometry("1180x800")
    #window.iconbitmap('image-test/Saki-Snowish-Traffic-light.ico')
    layout_()
    thread = Thread(target=load_model)
    #thread.start()
    #thrr = Thread(target=draw_chart)
    #thrr.start()
    laser_line_color = (0, 0, 255)
    video = cv2.VideoCapture('image-test/cam2.mp4')
    X = []
    Y = []
    continuePlotting = False
    btn_draw= False
    check_loadmd = False
    st= 0
    frame_count = 0
    car_number = 0
    obj_cnt = 0
    curr_trackers = []
    max_distance = 50
    update_frame()

    window.mainloop()