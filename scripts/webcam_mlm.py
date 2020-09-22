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

from ImageTransformer import ImageTransformer


def stylize_video(): #save_vid=False:

    ## Preparation for writing the ouput video
    #if save_vid:
    #    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #    out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))

    ##reading from the webcam
    cap = cv2.VideoCapture(0)

    times = [0, 0, 0]
    count = 0

    models_list = [
        "metzinger_bird", "kandinsky_composition", "delauney_rythme",
        "monet_blue", "Wadsworth_dazzleships", "gorky_artichoke",
        "vanDoesburg_CompositionI", "bruegel_babel", "gorky_liver",
        "taeuber-arp_composition", "cole_deluge"
    ]
    num_models = len(models_list)
    model_id = 0
    model_tail = "_linear_8.mlmodel"

    # Load torch model
    model_base = models_list[model_id]
    stored_file = model_base + model_tail
    print("Loading: " + stored_file)
    model = MLModel(stored_file)

    t0 = time.time()

    while cap.isOpened():
        ret, bgrimg = cap.read()
        if not ret:
            break
        count += 1

        t1 = time.time()
        bgrimg = cv2.resize(bgrimg, (320, 240))
        bgrimg = flip(bgrimg, axis=1)
        img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB).astype(np.float)
        img /= 255.0
        # convert to torch tensor
        img_t = torch.tensor(img).permute(2, 0, 1).to(torch.float)

        t2 = time.time()
        # stylize image
        sty = np.array(
            model.predict({"data": transforms.ToPILImage()(img_t)})['553'])[0]
        t3 = time.time()

        sty += 1.0
        sty *= 0.5
        sty = clip(np.moveaxis(sty, 0, -1), 0, 1)
        stybgr = cv2.cvtColor(sty, cv2.COLOR_RGB2BGR)

        cv2.imshow("video", stybgr)

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
        elif keypress == ord('a'):
            # a to load the previous model
            model_id -= 1
            if model_id < 0:
                model_id = num_models - 1
            model_base = models_list[model_id]
            stored_file = model_base + model_tail
            print("Loading: " + stored_file)
            model = MLModel(stored_file)

    print("fps = {} - prep {}, eval {}, post {}".format(
        count / (time.time() - t0), times[0], times[1], times[2]))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with torch.no_grad():
        stylize_video()
