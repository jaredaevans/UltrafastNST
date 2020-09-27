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

from ImageTransformer import ImageTransformer

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

    models_list = ["../../benches/bird_bench_Db", 
                   "../../benches/bird_bench_Db", 
                   "../../benches/bird_bench_Db", 
                   #"bird_bench_C", "bird_bench_D", 
        #"comp7_bench_B","comp7_bench_C","comp7_bench_D",
    ]
    num_models = len(models_list)
    model_id = 0
    model_tail = ".pth"

    # Load torch model
    model_base = models_list[model_id]
    stored_file = model_base + model_tail
    print("Loading: " + stored_file)
    model = ImageTransformer(leak=0.05,
                             norm_type='batch',
                             DWS=True,DWSFL=False,
                             outerK=3,resgroups=4,
                             filters=[8,12,16],
                             shuffle=True,
                             blocks=[1,2,1,2,1,2,1,1],
                             endgroups=(1,1),
                             bias_ll=True)
    model.load_state_dict(torch.load(stored_file, map_location=torch.device('cpu')))
    model.eval()
    
    t0 = time.time()
    
    while cap.isOpened():
        ret, bgrimg = cap.read()
        if not ret:
            break
        count += 1

        t1 = time.time()
        bgrimg = cv2.resize(bgrimg, (320, 240), interpolation=cv2.INTER_AREA)
        bgrimg = flip(bgrimg, axis=1)
        img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB).astype(np.float)
        img /= 127.5  # 0 - 255 to 0.0 - 2.0
        img -= 1 # 0.0 - 2.0  to -1.0 to 1.0
        # convert to torch tensor
        img_t = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(torch.float)

        t2 = time.time()
        # stylize image
        sty = model(img_t)
        t3 = time.time()

        sty = sty.squeeze().data.clamp_(-1, 1).permute(1,2,0).numpy()
        sty += 1.0
        sty *= 0.5
        sty = clip(sty,0,1)
        stybgr = cv2.cvtColor(sty, cv2.COLOR_RGB2BGR)

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
            model.load_state_dict(torch.load(stored_file, map_location=torch.device('cpu')))
        elif keypress == ord('a'):
            # a to load the previous model
            model_id -= 1
            if model_id < 0:
                model_id = num_models - 1
            model_base = models_list[model_id]
            stored_file = model_base + model_tail
            print("Loading: " + stored_file)
            model.load_state_dict(torch.load(stored_file, map_location=torch.device('cpu')))

    print("fps = {} - prep {}, eval {}, post {}".format(
        count / (time.time() - t0), times[0], times[1], times[2]))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with torch.no_grad():
        stylize_video()
