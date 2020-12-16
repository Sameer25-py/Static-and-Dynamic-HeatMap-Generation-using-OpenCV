import cv2
import numpy as np 
import json
import matplotlib.pyplot as plt
import threading


def read_boxes(file):
    points_per_frame = []
    with open(file,'r') as f:
        data = json.load(f)
    f.close()
    for frame in data:
        temp = []
        for frame_data in frame:
            dic = {}
            dic['class'] = frame_data['class']
            dic['pointx'] = frame_data['start_X'] + (frame_data['width']/2)
            dic['pointy'] = frame_data['start_Y'] + (frame_data['height']/2)
            temp.append(dic)
        points_per_frame.append(temp)
    return points_per_frame

def read_top_boxes(file):
    points_per_frame = []
    with open(file,'r') as f:
        data = json.load(f)
    f.close()
    for frame in data:
        temp = []
        for frame_data in frame:
            dic = {}
            dic['class'] = frame_data['class']
            dic['pointx'] = frame_data['pointx']
            dic['pointy'] = frame_data['pointy']
            temp.append(dic)
        points_per_frame.append(temp)
    return points_per_frame


def generate_kernel(ksize,std):
    #ksize = 27
    std = 0.3*((ksize-1)*0.5 - 1) + 0.8
    x = cv2.getGaussianKernel(ksize,std)
    x = (x*x.T)
    x=x/np.max(x)
    return x

def generate_colormap(x):
    colored = cv2.applyColorMap((x*255).astype(np.uint8),cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return colored

def generate_heatmap(t_points,img,colored,ksize):
    for points in t_points:
        for i in points:
            if i['pointy']+int(ksize/2) >= 300 or i['pointx'] + int(ksize/2) >= 600 or i['pointy'] - int(ksize/2) < 0 or i['pointx'] - int(ksize/2) < 0:
                continue
            point_on_img = img[int(i['pointy'])-int(ksize/2):int(i['pointy'])+ int(ksize/2) + 1,int(i['pointx'])-int(ksize/2):int(i['pointx'])+ int(ksize/2) + 1,:]
            new_points =(point_on_img + colored)
            img[int(i['pointy'])-int(ksize/2):int(i['pointy'])+ int(ksize/2) + 1,int(i['pointx'])-int(ksize/2):int(i['pointx'])+ int(ksize/2) + 1,:] = new_points
    img /= img.max(axis=(0,1)) + 0.1
    return img

def generate_inv_mask(img,frame):
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    inv_mask = np.where(gray_img > 0.7,0.0,1.0)
    inv_mask3d = np.ones((height,width,3),dtype=np.float32) * ((inv_mask)[:,:,None])
    return frame*inv_mask3d

def capture(camera,points,colored,th_step,name):
    global sync
    frame_counter = 0
    while True:
        ret,frame = camera.read()
        if not ret:
            break
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)/255
        frame = cv2.resize(frame,(width,height))/255.0
        plain_black_img = np.zeros((height,width,3),dtype=np.float32)
        heated = []
        if sync == False:
            break
        if frame_counter < len(points):
            if th_step == 0:
                heated  = generate_heatmap(points[:frame_counter+1],plain_black_img,colored,ksize)
            else:
                if frame_counter < th_step:
                    heated  = generate_heatmap(points[:frame_counter+1],plain_black_img,colored,ksize)
                else:
                    heated  = generate_heatmap(points[frame_counter-th_step-1:frame_counter],plain_black_img,colored,ksize)
            mask = generate_inv_mask(heated,frame)
            frame = mask + heated
        cv2.imshow(name,frame)
        if cv2.waitKey(33) & 0xff == ord('q'):
            sync = False
            break
        frame_counter+=1
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sync = True
    width = 600
    height = 300
    ksize = 21
    th_step = 10
    img = np.zeros((height,width,3),dtype=np.float32)
    point_files = ['boxes/camera0.txt','boxes/camera1.txt','boxes/camera2_top.txt']
    video_files = ['movies/cam0.avi','movies/cam1.avi','movies/cam2_top.avi']
    t_points=[]
    cameras = []
    for i in range(len(point_files)-1):
        t_points.append(read_boxes(point_files[i]))
    t_points.append(read_top_boxes(point_files[-1]))
    for cam in video_files:
        cameras.append(cv2.VideoCapture(cam))
    kernel = generate_kernel(ksize,10)
    colored = generate_colormap(kernel)
    threads = []
    for i in range(len(cameras)):
        t1 = threading.Thread(target=capture,args=(cameras[i],t_points[i],colored,th_step,'camera' + str(i)),daemon = True)
        threads.append(t1)
        t1.start()
    for i in threads:
        i.join()
    cv2.destroyAllWindows()




    