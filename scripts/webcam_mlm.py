#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Python program to access the webcam, stream video, modify it via
    the image transformer, and display the output in a window 
"""

import time
import cv2
import torch
import numpy as np
from numpy import flip, clip

import PIL
from PIL import Image
import torchvision.transforms as transforms
""" Note: need to import unused onnx or coreml crashes program
    bug reported submitted to coreml
"""
import onnx
from coremltools.models import MLModel

#from ImageTransformer import ImageTransformer

def layertag(idkey):
    if idkey=='B' or idkey=='C' or idkey=='D':
        return '983'
    elif idkey=='b':
        return '1058'
    else:
        return '974'

def stylize_video(): #save_vid=False:

    ## Preparation for writing the ouput video
    #if save_vid:
    #    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #    out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))

    ##reading from the webcam
    cap = cv2.VideoCapture(0)

    times = [0, 0, 0]
    count = 0
    
    # fps text info
    font = cv2.FONT_HERSHEY_TRIPLEX
    org = (220, 230) 
    fontScale = 0.5
    color = (0, 0.5, 0.8) 

    models_list = ["bird_bench_Db", 
                   "scream_bench_D", 
                   "jazzcups_D",
                   "dazzleships_D",
                   "delauney_rythme_D",
                   "monet_blue_D",
                   "scream_bench_B", 
                   "scream_bench_C", #"scream_bench_D",  
                   "bird_bench_B",
                   "scream_bench_I", 
                   "bird_bench_I", 
                   "comp7_bench_I",
                   "scream_bench_J", 
                   "bird_bench_J", 
                   "comp7_bench_J", 
                   #"bird_bench_C", "bird_bench_D", 
        #"comp7_bench_B","comp7_bench_C","comp7_bench_D",
    ]
    num_models = len(models_list)
    model_id = 0
    model_tail = "_linear_16.mlmodel"

    # Load torch model
    model_base = models_list[model_id]
    stored_file = model_base + model_tail
    print("Loading: " + stored_file)
    model = MLModel(stored_file)

    timelist = []
    
    t0 = time.time()
    timelist.append(t0)
    
    idkey = model_base[-1]
    layerval = layertag(idkey)

    while cap.isOpened():
        ret, bgrimg = cap.read()
        if not ret:
            break
        count += 1

        t1 = time.time()
        bgrimg = cv2.resize(bgrimg, (320, 240), interpolation=cv2.INTER_AREA)
        bgrimg = flip(bgrimg, axis=1)
        img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB).astype(np.float)
        img /= 255.0
        # convert to torch tensor
        img_t = torch.tensor(img).permute(2, 0, 1).to(torch.float)

        t2 = time.time()
        # stylize image
        sty = np.array(
            model.predict({"data": transforms.ToPILImage()(img_t)})[layerval])[0]
        t3 = time.time()
        timelist.append(t3)

        sty += 1.0
        sty *= 0.5
        sty = clip(np.moveaxis(sty, 0, -1), 0, 1)
        stybgr = cv2.cvtColor(sty, cv2.COLOR_RGB2BGR)
        nf = len(timelist)
        if nf > 3:
            text = "fps: {:0.4g}".format(nf/(timelist[-1]-timelist[0]))
            stybgr = cv2.putText(stybgr,text,org,font,fontScale,color) 
            if nf > 10:
                timelist = timelist[1:]

        cv2.imshow("video", cv2.resize(stybgr,(640,480), 
                                       interpolation=cv2.INTER_LINEAR))

        times[0] += t2 - t1
        times[1] += t3 - t2
        times[2] += time.time() - t3

        keypress = cv2.waitKey(1)
        if keypress == ord('q'):
            # q to quit
            break
        elif keypress == ord('d') or keypress == ord('r'):
            # r or d to load the next model
            model_id += 1
            if model_id >= num_models:
                model_id = 0
            model_base = models_list[model_id]
            stored_file = model_base + model_tail
            print("Loading: " + stored_file)
            model = MLModel(stored_file)
            idkey = model_base[-1]
            layerval = layertag(idkey)
        elif keypress == ord('a'):
            # a to load the previous model
            model_id -= 1
            if model_id < 0:
                model_id = num_models - 1
            model_base = models_list[model_id]
            stored_file = model_base + model_tail
            print("Loading: " + stored_file)
            model = MLModel(stored_file)
            idkey = model_base[-1]
            layerval = layertag(idkey)

    print("fps = {} - prep {}, eval {}, post {}".format(
        count / (time.time() - t0), times[0], times[1], times[2]))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with torch.no_grad():
        stylize_video()
