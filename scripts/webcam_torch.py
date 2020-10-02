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

def build_model(idkey):
    """
    Default parameters:
    ImageTransformer(leak=0.05,
                    norm_type='batch',
                    DWS=True,
                    DWSFL=False,
                    outerK=3,
                    resgroups=4,
                    filters=[8,12,16],
                    shuffle=True,
                    blocks=[1,2,1,2,1,2,1,1],
                    endgroups=(1,1),
                    upkern=4,
                    bias_ll=True)

    """
    if idkey=='V':
        model = ImageTransformer()
    model.eval()
    model.fuse()
    return model
    
def quantize_model(image_transformer):
    """ Convert model to quantized version """
    image_transformer.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # insert observers
    torch.quantization.prepare(image_transformer, inplace=True)
    # Calibrate the model and collect statistics
    # convert to quantized version
    torch.quantization.convert(image_transformer, inplace=True)

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
    
    isHalf = False
    
    dir_path = "../models/"


    ''' "bruegel_babel_", 
    "bosch_tondal_",
    "delaunay_window_", 
    "vanDoesburg_CompositionI_", 
    
    "gorky_liver_" '''

    bench_list = [
        "scream_bench_", "bird_bench_", "comp7_bench_",
         "Jazzcups_","dazzleships_", "comp7_", "monet_blue_", 
         "gorky_artichoke_", "delauney_rythme_"
    ]
    num_benches = len(bench_list)
    bench_id = 0
    
    models_list = ["V"]
    num_models = len(models_list)
    model_id = 0
    
    model_tail = ".pth"

    # Load torch model
    bench_base = bench_list[bench_id]
    model_base = models_list[model_id]
    stored_file = dir_path + bench_base + model_base + model_tail
    print("Loading: " + stored_file)
    
    model = build_model(model_base)
    model.load_state_dict(torch.load(stored_file, map_location=torch.device('cpu')))
    model.eval()
    
    timelist = []
    
    t0 = time.time()
    timelist.append(t0)
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
        if isHalf:
            dtype = torch.float16
        else:
            dtype = torch.float
        img_t = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(dtype)

        t2 = time.time()
        # stylize image
        sty = model(img_t)
        t3 = time.time()
        timelist.append(t3)

        sty = sty.squeeze().data.clamp_(-1, 1).permute(1,2,0).numpy()
        sty += 1.0
        sty *= 0.5
        sty = clip(sty,0,1)
        stybgr = cv2.cvtColor(sty, cv2.COLOR_RGB2BGR)
        
        nf = len(timelist)
        if nf > 3:
            text = "fps: {:0.3g}".format((nf-1)/(timelist[-1]-timelist[0]))
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
            stored_file = dir_path + bench_base + model_base + model_tail
            print("Loading: " + stored_file)
            model = build_model(model_base)
            if model_base[-1] == "q":
                quantize_model(model)
            elif model_base[-1] == "h":
                model.half()
                isHalf = True
            else:
                isHalf = False
            model.load_state_dict(torch.load(stored_file, map_location=torch.device('cpu')))
            model.eval()    
        elif keypress == ord('a'):
            # a to load the previous model
            model_id -= 1
            if model_id < 0:
                model_id = num_models - 1
            model_base = models_list[model_id]
            stored_file = dir_path + bench_base + model_base + model_tail
            print("Loading: " + stored_file)
            model = build_model(model_base)
            if model_base[-1] == "q":
                quantize_model(model)
            elif model_base[-1] == "h":
                model.half()
                isHalf = True
            else:
                isHalf = False
            model.load_state_dict(torch.load(stored_file, map_location=torch.device('cpu')))
            model.eval() 
        elif keypress == ord('w'):
            bench_id += 1 
            if bench_id >= num_benches:
                bench_id = 0
            bench_base = bench_list[bench_id]
            stored_file = dir_path + bench_base + model_base + model_tail
            print("Loading: " + stored_file)
            model.load_state_dict(torch.load(stored_file, map_location=torch.device('cpu')))
            model.eval()
        elif keypress == ord('s'):
            bench_id -= 1 
            if bench_id < 0:
                bench_id = num_benches - 1
            bench_base = bench_list[bench_id]
            stored_file = dir_path + bench_base + model_base + model_tail
            print("Loading: " + stored_file)
            model.load_state_dict(torch.load(stored_file, map_location=torch.device('cpu')))
            model.eval() 
    print("fps = {} - prep {}, eval {}, post {}".format(
        count / (time.time() - t0), times[0], times[1], times[2]))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with torch.no_grad():
        stylize_video()
