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
from portraitsegmenter import PortraitSegmenter

def build_model(idkey):
    """
    Default parameters:
    ImageTransformer(leak=0,
                     norm_type='batch',
                     DWS=True,DWSFL=False,
                     outerK=3,resgroups=1,
                     filters=[8,16,16],
                     shuffle=False,
                     blocks=[2,2,2,1,1],
                     endgroups=(1,1),
                     upkern=3,
                     bias_ll=True,
                     quant=False)

    """
    label = "Artiscope"
    if idkey=='V' or idkey=='Z':
        model = ImageTransformer(leak=0,
                            norm_type='batch',
                            DWS=True,DWSFL=False,
                            outerK=3,resgroups=1,
                            filters=[8,16,16],
                            shuffle=False,
                            blocks=[2,2,2,1,1],
                            endgroups=(1,1),
                            upkern=3,
                            bias_ll=True)
        model.eval()
        model.fuse()
        label = "Artiscope"
    elif idkey=='A':
        model = ImageTransformer(norm_type='batch',
                                DWS=False,
                                outerK=9,
                                resgroups=1,
                                filters=[32,64,128],
                                shuffle=False,
                                blocks=[1,1,1,1,1])
        model.eval()
        label = "Johnson et al"
    elif idkey=='C':
        model = ImageTransformer(norm_type='inst',
                                DWS=False,
                                outerK=3,
                                resgroups=1,
                                filters=[16,24,32],
                                shuffle=False,
                                blocks=[1,1,1,1])
        model.eval()
        label = "Kunster"
    return model, label

def build_seg_model():
    """
    port_seg = PortraitSegmenter(down_depth=[1,2,2,2], up_depth=[1,1,1],
                                 filters=[16,24,32,48])
    stored_file = "../models/portraitCE.pth"
    """
    port_seg = PortraitSegmenter(down_depth=[1,2,2], num_levels=3, up_depth=[1,1],
                 filters=[16,24,32],endchannels=[8,1])
    stored_file = "../models/portraitCElight.pth"
    port_seg.load_state_dict(torch.load(stored_file, map_location=torch.device('cpu')))
    port_seg.eval()
    port_seg.fuse()
    port_seg.eval()
    return port_seg
    
def quantize_model(image_transformer):
    """ Convert model to quantized version """
    image_transformer.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # insert observers
    torch.quantization.prepare(image_transformer, inplace=True)
    # Calibrate the model and collect statistics
    # convert to quantized version
    torch.quantization.convert(image_transformer, inplace=True)
    
def convert_mask(mask, thresh=1.65):
    new = cv2.inRange(mask, thresh, 10000)
    return new

def stylize_video(save_vid=False):

    ## Preparation for writing the ouput video
    if save_vid:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vidout = cv2.VideoWriter('output.avi', fourcc, 10.0, (640, 480))
    isRecording = False

    ##reading from the webcam
    cap = cv2.VideoCapture(0)

    times = [0, 0, 0, 0, 0]
    count = 0
    pic_count = 0
    
    # common text info
    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    
    # fps text info
    fps_org = (440, 460) 
    fps_color = (0, 0.0, 0.8) 
    
    # label text info
    label_org = (10,30)
    label_color = (0, 0.0, 0.8)
    
    isHalf = False
    
    dir_path = "../models/"

    bench_list = [
        "scream_bench_", "bird_bench_", "comp7_bench_", "comp7_",
        "dazzleships_",  "monet_blue_", "gorky_artichoke_", 
        "delauney_rythme_","taeuber-arp_composition_",
        "vanDoesburg_CompositionI_", "bruegel_babel_", "delaunay_window_",
        "bosch_tondal_","jazzcups_", 
    ]
    num_benches = len(bench_list)
    bench_id = 0
    
    models_list = ['V','Z','A'] #'C','A']
    num_models = len(models_list)
    model_id = 0
    
    model_tail = ".pth"

    # Load torch model
    bench_base = bench_list[bench_id]
    model_base = models_list[model_id]
    stored_file = dir_path + bench_base + model_base + model_tail
    print("Loading: " + stored_file)
    
    model, label = build_model(model_base)
    model.load_state_dict(torch.load(stored_file, map_location=torch.device('cpu')))
    model.eval()
    
    will_segment = False
    segment_invert = False
    maskThresh = 2.0
    segmenter = build_seg_model()
    
    ellipse1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cross1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10))
    
    timelist = []
    
    t0 = time.time()
    timelist.append(t0)
    while cap.isOpened():
        ret, bgrimg = cap.read()
        if not ret:
            break
        count += 1

        t1 = time.time()
        bgrimg_full = cv2.resize(bgrimg, (640, 480), interpolation=cv2.INTER_AREA)
        bgrimg_full = flip(bgrimg_full, axis=1)
        bgrimg = cv2.resize(bgrimg_full, (320, 240), interpolation=cv2.INTER_AREA)
        #bgrimg = flip(bgrimg, axis=1)
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
        
        stybgr = cv2.resize(stybgr,(640,480),interpolation=cv2.INTER_LINEAR)
        
        t4 = time.time()
        t5 = time.time()
        if will_segment:
            seg_input = cv2.resize(img, (64, 48), interpolation=cv2.INTER_AREA)
            mask, edge = segmenter(torch.tensor(seg_input).permute(2,0,1).unsqueeze(0).to(dtype))
            t5 = time.time()
            mask1 = convert_mask(mask[0].permute(1,2,0).numpy(),maskThresh)
            
            mask1 = cv2.resize(mask1, (640, 480), interpolation=cv2.INTER_LINEAR)
            #mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, ellipse1)
            #mask1 = cv2.morphologyEx(mask1, cv2.MORPH_ERODE, ellipse2)
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, cross1)
            
            mask1 = convert_mask(mask1,maskThresh)
            
            # Create an inverted mask
            mask2 = cv2.bitwise_not(mask1)
            
            stybgr = (stybgr*255).astype(np.uint8)
            if segment_invert:
                # Segment the stylized portrait
                res1 = cv2.bitwise_and(stybgr, stybgr, mask=mask1)
            
                # Segment the true image
                res2 = cv2.bitwise_and(bgrimg_full, bgrimg_full, mask=mask2)
            else:
                # Segment the true portrait
                res1 = cv2.bitwise_and(stybgr, stybgr, mask=mask2)
            
                # Segment the sylized background
                res2 = cv2.bitwise_and(bgrimg_full, bgrimg_full, mask=mask1)
            
            # Generating the final output and writing
            stybgr = cv2.addWeighted(res1, 1, res2, 1, 0)
            stybgr = stybgr.astype(np.float)/255.
            
        
        nf = len(timelist)
        if nf > 3:
            text = "fps: {:0.3g}".format((nf-1)/(timelist[-1]-timelist[0]))
            stybgr = cv2.putText(stybgr,text,fps_org,font,fontScale,fps_color)
            stybgr = cv2.putText(stybgr,label,label_org,font,fontScale,label_color) 
            if nf > 10:
                timelist = timelist[1:]
        
        if save_vid and isRecording:
            vidout.write(np.uint8(stybgr*255))
        cv2.imshow("video", stybgr)
        
        times[0] += t2 - t1
        times[1] += t3 - t2
        times[2] += t4 - t3
        times[3] += t5 - t4
        times[4] += time.time() - t5

        keypress = cv2.waitKey(1)
        if keypress == ord('q'):
            # q to quit
            break
        elif keypress == ord('d'):
            # d to load the next model
            model_id += 1
            if model_id >= num_models:
                model_id = 0
            model_base = models_list[model_id]
            stored_file = dir_path + bench_base + model_base + model_tail
            print("Loading: " + stored_file)
            model, label = build_model(model_base)
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
            model, label = build_model(model_base)
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
        elif keypress == ord('e'):
            will_segment = not will_segment
        elif keypress == ord('r'):
            segment_invert = not segment_invert 
        elif keypress == ord('x'):
            maskThresh *= 0.9
        elif keypress == ord('c'):
            maskThresh *= 1.1
        elif keypress == ord('g'):
            filename = "out_im_{}.jpg".format(pic_count)
            print("Saving frame as {}".format(filename))
            cv2.imwrite(filename,stybgr*255) 
            pic_count += 1
        elif keypress == ord('t'):
            isRecording = not isRecording
    print("fps = {} - prep {}, eval {}, post {}, seg {}, postseg {}".format(
        count / (time.time()-t0),times[0],times[1],times[2],times[3],times[4]))
    cap.release()
    if save_vid:
        vidout.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with torch.no_grad():
        stylize_video(save_vid=False)
