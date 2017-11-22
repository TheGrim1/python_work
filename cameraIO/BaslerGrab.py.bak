# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time
import pypylon as py
import matplotlib.pyplot as plt
import numpy as np
import cv2

def initialize_cameras():
    '''
    if cameras are not properly closed reinitilize
    '''
    cameras = []
    available_cameras = py.factory.find_devices()
    for i in range(len(available_cameras)):
        print available_cameras[i]
        try:
            cameras.append(py.factory.create_device(available_cameras[i]))
        except Exception:
            print 'error opening camera:'     
            print Exception.args
            print Exception.message
           

        
    print "%s cameras ready" % len(cameras)
    return cameras

def grab_image(cam, bw = False):
    '''
    grabs an image from the given camera. 
    the camera must have been initialized. No other cameras can be open.
    '''   

    if bw == False:
        print 'pypylon only supports mono8 ATM' 
    cam.open()
    image = cam.grab_image()

    cam.close()

    return image

def grab_images(cameras, bw = False):
    '''
    grabs an imaga (grab_image) from each camera in the list.
    returns list of images (numpy arrays)
    '''
    images = []
    for cam_no, cam in enumerate(cameras):
        images.append(grab_image(cam, bw))
            
    
    return images


def identify_cameras(cameras = None):
    '''
    plots images from cameras to identify them
    '''
    if cameras == None:
        cameras = initialize_cameras()
    
    for cam_no, cam in enumerate(cameras):
        fig0, ax0 = plt.subplots(1)
        image0    = grab_image(cam)
        ax0.imshow(image0)
        ax0.set_title('camera %s'%cam_no)
    
    return

def liveview(camera, bw = False):
    try: 
        plt.ion()
        i = 0
        while True:
            image = grab_image(camera, bw = bw)
            i+=1
            plt.imshow(image)
            print 'showing frame %s'%i  
            plt.pause(0.05)     
    
    except Exception:
        print 'live view ended'
        print Exception.args
        print Exception.message
        plt.ioff()
        pass



def plot_cameras_images(cameras):
    '''
    test function to plt an image from the list of cameras given
    '''
    for cam_no, cam in enumerate(cameras):
        image = grab_image(cam)  
        print 'camera number %s' %cam_no
        plt.matshow(image.sum(-1))
        plt.show()
        
def time_grabbing(no_frames):
    '''
    test function to time the image grabbing
    '''
    start_time = time.time()
    cameras = initialize_cameras()
    open_time = time.time() - start_time
    print "opening cameras took %s s" % open_time
        
    for i in range(no_frames):
        grab_images(cameras)    
        
    grab_time = time.time() - start_time - open_time

    print 'grabbing %s images on %s cameras took %s s' % \
        (no_frames, len(cameras), grab_time )
        
#    for i in range(no_frames):
#        plot_cameras_images(cameras)
#    plot_time = time.time() - start_time - grab_time - open_time
#    print 'grabbing and plotting %s images on %s cameras took %s s' % \
#        (no_frames, len(cameras), plot_time )

def test():
    cameras = initialize_cameras()
    plot_cameras_images(cameras)
    
if __name__=='__main__':
    test()