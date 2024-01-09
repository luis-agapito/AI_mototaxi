#!/usr/bin/env python
# coding: utf-8

import collections
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy
import os
import pickle
import PIL
import sys

matplotlib.use('Qt5agg')

source_dir = '~/Downloads/dldata/mototaxi_binary/1920x1920/no_mototaxi/'
dest_dir = '~/Downloads/dldata/mototaxi_binary/1920x1920/no_mototaxi_croppings/'
maxX, maxY = 1920, 1920

source_dir = '~/Downloads/dldata/mototaxi_binary/1536x2560/no_mototaxi/'
dest_dir = '~/Downloads/dldata/mototaxi_binary/1536x2560/no_mototaxi_croppings/'
maxX, maxY = 1536, 2560

dRmin = 224
corner_thr = 100 #Cutoff to detect whether we are selecting a special point (corners, center)
dR_same_thr = 150
maxR = min(maxX, maxY)

counter = 0

Point = collections.namedtuple('Point', ['x', 'y'])

def writeCropping(x, y, W, H, im, dest_dir, file_jpg, i):
    aux1, aux2 = os.path.splitext(os.path.join(dest_dir, file_jpg))
    file_jpg_full = aux1 + '_' + str(i) + aux2
    x_topleft, y_topleft = x, y
    x_bottomright, y_bottomright = x + W, y + H
    cropped_im = im.crop((x_topleft, y_topleft, x_bottomright, y_bottomright))
    cropped_im.save(file_jpg_full)

# write list to binary file
def write_list(a_list, source_dir, list_name):
    # store list in binary file so 'wb' mode
    with open(os.path.join(source_dir, list_name), 'wb') as fp:
        pickle.dump(a_list, fp)
        #print('Done writing list into a binary file')

def read_list(source_dir, list_name):
    # for reading also binary mode is important
    with open(os.path.join(source_dir, list_name), 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

def getNextInFileList(dest_dir, list_name, source_dir):
    file_list = read_list(dest_dir, list_name)
    if len(file_list)==0 :
        #sys.exit('Finished processing all files to be processed. Exiting.')
        return 0, None 
    aux_file = os.path.join(source_dir, file_list[0])
    print('get_next_in_file_list = ', aux_file)
    return len(file_list), aux_file

def update_file_list(dest_dir, list_name, current_file_name):
    file_list = read_list(dest_dir, list_name)
    assert(current_file_name == file_list[0])
    file_list.pop(0)
    write_list(file_list, dest_dir, list_name)

def writePartialListOfFiles(file_jpg):
    """
    Recreates a json file containing all filenames, starting at the give file 'file_jpg'.
    :param file_jpg:
    :return:
    """
    if file_jpg=="":
        found=True
    else:
        found = False
    file_list = []
    for f in sorted(os.listdir(source_dir)):
        if f.startswith('.'): continue
        if f==file_jpg:
            found = True
            continue
        if found==True:
            file_list.append(f)

    print(file_list)
    print(len(file_list))
    write_list(file_list, dest_dir, 'files_unprocessed')

def isNear(point, ref_point):
    return True if (abs(point.x - ref_point.x) < corner_thr
                    and 
                    (abs(point.y - ref_point.y) < corner_thr)
                    ) else False
    
def onselect(eclick, erelease):
    global im, file_jpg_full, dest_dir
    if eclick.ydata>erelease.ydata:
        eclick.ydata,erelease.ydata=int(erelease.ydata), int(eclick.ydata)
    if eclick.xdata>erelease.xdata:
        eclick.xdata,erelease.xdata=int(erelease.xdata), int(eclick.xdata)

    dX = erelease.xdata - eclick.xdata
    dY = erelease.ydata - eclick.ydata
    dR = max( int((dX+dY)/2), dRmin)
    click = Point(x=eclick.xdata, y=eclick.ydata)
    release = Point(x=erelease.xdata, y=erelease.ydata)
    #print(click, release)
    if isNear(click, release):
        maxX, maxY = im.size
        dR = maxR
        avg_r = Point(x=int((eclick.xdata + erelease.xdata)/2), y=int((eclick.ydata + erelease.ydata)/2))
        if isNear(avg_r, Point(x=0.0, y=0.0)):
            # top left
            eclick.xdata, eclick.ydata = 0.0, 0.0
            erelease.xdata, erelease.ydata = maxR, maxR
            #print("top left")
        elif isNear(avg_r, Point(x=maxX, y=0.0)):
            # top right
            eclick.xdata, eclick.ydata = maxX - maxR, 0.0
            erelease.xdata, erelease.ydata = maxX, maxR
            #print("top right ")
        elif isNear(avg_r, Point(x=0.0, y=maxY)):
            # bottom left
            eclick.xdata, eclick.ydata = 0.0, maxY - maxR
            erelease.xdata, erelease.ydata = maxR, maxY
            #print("bottom left")
        elif isNear(avg_r, Point(x=maxX, y=maxY)):
            # bottom right
            eclick.xdata, eclick.ydata = maxX-maxR, maxY-maxR
            erelease.xdata, erelease.ydata = maxX, maxY
            #print(eclick, erelease)
            #print("bottom right")
        else:
            eclick.xdata, eclick.ydata = int(maxX/2 - maxR/2), int(maxY/2 - maxR/2)
            erelease.xdata, erelease.ydata = int(maxX/2 + maxR/2), int(maxY/2 + maxR/2)
            #print("center")
    else:
        pass
        #print("custom")

    nold = len(ax.patches)
    rect = matplotlib.patches.Rectangle((eclick.xdata, eclick.ydata), dR, dR,linewidth=2,edgecolor='r',facecolor='none')
    #nold = len(ax.patches)
    p = ax.add_patch(rect)
    print('---------------------')
    import subprocess
    app_name = "iTerm2"
    cmd = f'osascript -e \'activate application "{app_name}"\''
    subprocess.call(cmd, shell=True)

    answer = input("Accept(y) / Reject(n) / Accept&Finish(h) / Finish (j) ")
    #keep_looping = True
    while answer not in ['y', 'n', 'h', 'j']:
        answer = input("Accept(y)/Reject(n)/Accept&Finish(h)/Finish(j)")

    print(answer)
    if answer=="n":
        p.remove()
    elif answer=="h" or answer=="j":
        path, file_jpg = os.path.split(file_jpg_full)
        if answer=="h":
            for i, p in enumerate(ax.patches):
                if i==0: continue
                x, y, W, H = int(p.get_x()), int(p.get_y()), p.get_width(), p.get_height()
                #im_array = numpy.array()
                #print(i, 'patches x,y,W,H', p.get_x(), p.get_y(), p.get_width(), p.get_height())
                p.remove()
                writeCropping(x, y, W, H, im, dest_dir, file_jpg, i)
        if answer=="j":
            p.remove()
        update_file_list(dest_dir, 'files_unprocessed', file_jpg) #uses 'file_jpg' for assertion.
        nleft, file_jpg_full = getNextInFileList(dest_dir, 'files_unprocessed', source_dir)
        if nleft==0:
            sys.exit('Finished processing all files in source directory. Exiting.')
        #-----------------------

        aux = [f for f in sorted(os.listdir(dest_dir)) if not f.startswith('.')]
        print('Number of files in dest_dir = ', len(aux))
        #
        _, aux = os.path.split(file_jpg_full)
        plt.title(str(nleft) + '; ' +aux)
        im = PIL.Image.open(file_jpg_full)
        #im = PIL.ImageOps.exif_transpose(im)
        plt_image=plt.imshow(im, origin="upper")
        w, h = im.width, im.height
        plt.autoscale(False)
        plt.plot(w/2, h/2, "og")
        #plt.plot([0, w], [0, h], ":k")
        #plt.plot([0, w], [h, 0], ":k")
        plt.ylim(h, 0) #b/c origin=="upper"
        plt.xlim(0, w)

        fig.canvas.draw()

if __name__ == "__main__":
    #plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    #file_jpg = '20231221_142108.jpg' #all files after this will be included in the new list
    file_jpg = '' #all files after this will be included in the new list
    writePartialListOfFiles(file_jpg)

    nleft, file_jpg_full = getNextInFileList(dest_dir, 'files_unprocessed', source_dir)
    im = PIL.Image.open(file_jpg_full)
    import PIL.ImageOps as imageops
    im = imageops.exif_transpose(im)
    #arr = np.asarray(im)
    _, aux = os.path.split(file_jpg_full)
    plt.title(str(nleft) + '; ' + aux)
    plt_image=plt.imshow(im) #, origin="lower")
    rs=widgets.RectangleSelector(ax, onselect, #drawtype='box',
        props = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
    plt.show()
