# UltrafastNST
A ultrafast neural network to perform style-transfer on real-time video over a CPU.  This github is associated with this [blog post](https://medium.com/@jaevans_98274/ultrafast-neural-style-transer-on-a-cpu-a8b1c7e7fc8b).

## Requirements:
- Clone repository
- PyTorch>=1.6.0 
- OpenCV>=4.4.0
- Note: Windows users may need to follow [these instructions to install pytorch](https://stackoverflow.com/questions/47754749/how-to-install-pytorch-in-windows)

## To run:
./webcam_torch.py

Press:
- w/s to change styles
- a to change to a heavier/lighter stylization 
- e to segment yourself out of the image
- r to invert the mask
- q to quit

## To train new models:
Create_Model.ipynb has most of what you need, I recommend running on Colab.  First, upload the requisite scripts (basically all of them).  Second, upload many images - the specific dataset is not particularly important (e.g., MS COCO or Flicker8k).  Third, pick a style, and train!  The equalized stylization loss should help with getting a good quality.  If it is over (under) stylized, decrease (increase) the style weight.

## License:
MIT

