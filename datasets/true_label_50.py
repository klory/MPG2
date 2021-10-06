import math
import torch

labels = {}

def label_dict(top, bottom,left,right, base, size = 1080,rotation = 0):
    dct ={"top":top,"bottom":bottom,"left":left,"right":right, 'base': base, "size":size,"rotation":rotation}
    return dct
def label_dict_center(top,left,width,height, base, size = 1080,rotation = 0):
    top = top-height/2
    bottom = size-top-height
    left = left-width/2
    right = size-left-width
    dct ={"top":top,"bottom":bottom ,"left":left,"right":right, 'base': base, "size":size,"rotation":rotation}
    return dct
# distance to top, bottom, left right
labels["00004"]=label_dict(75,25,-25,70, 0)
labels["00010"]=label_dict(100,70,-150,-150, 1)
labels["00015"]=label_dict(130,130,15,15, 0)
labels["00020"]=label_dict(60,95,5,25,0)
labels["00031"]=label_dict(280,0,0,-40, 1)
labels["00036"]=label_dict(25,45,30,15, 0)
labels["00041"]=label_dict(0,20,30,0, 1)
labels["00048"]=label_dict(95,125,-150,-100, 1)
# labels["00049"]=label_dict_center(270,755,950*2,850*2)
# labels["00060"]=label_dict_center(390,550,600*2,667*2)
# labels["00069"]=label_dict_center(780,400,800*2,660)
labels["00130"]=label_dict(130,90,-5,0, 1)
labels["00134"]=label_dict(50,70,25,30, 0)
labels["00141"]=label_dict(80,70,50,50, 1)
labels["00143"]=label_dict(60,25,0,90, 1)
labels["00144"]=label_dict(70,115,0,0, 0)
labels["00145"]=label_dict(13,10,10,15, 0, size=774)
labels["00147"]=label_dict(300,120,185,-50, 0)
labels["00148"]=label_dict(90,85,80,75, 1, size=655)
labels["00151"]=label_dict(80,200,115,50, 1)
labels["00156"]=label_dict(100,280,75,70, 1)
labels["00163"]=label_dict(35,35,15,15, 1, size=745)
# labels["00173"]=label_dict_center(570,50,675*2,900)
labels["00184"]=label_dict(165,-5,-105,105, 0)
labels["00198"]=label_dict(132,72,80,100, 0)
labels["00199"]=label_dict(167,210,65,65, 1)
labels["00204"]=label_dict(145,90,10,0, 1)
labels["00205"]=label_dict(175,90,-275,270, 1)
labels["00212"]=label_dict(90,35,30,60, 1)
labels["00217"]=label_dict(90,60,10,65, 1)
labels["00222"]=label_dict(370,120,25,-5, 1)
labels["00233"]=label_dict(5,60,40,0, 1)
labels["00235"]=label_dict(70,118,0,0, 0)
labels["00259"]=label_dict(110,20,20,25, 1)
labels["00264"]=label_dict(300,120,185,-50, 0)
labels["00272"]=label_dict(180,150,135,90, 0)
labels["00285"]=label_dict(450,27,250,-250, 0)
labels["00306"]=label_dict(160,100,75,115, 1)
labels["00429"]=label_dict(107,177,130,120, 0)
labels["00474"]=label_dict_center(745,855,580*2,560*2, 0)
labels["00480"]=label_dict(115,185,50,50, 1)
labels["00488"]=label_dict(110,95,30,25, 0)
labels["00531"]=label_dict(275,98,50,25, 0)
labels["00543"]=label_dict(414,0,195,220, 0)
labels["00546"]=label_dict(60,15,5,10, 0)
labels["00560"]=label_dict(80,85,5,95, 1)
labels["00564"]=label_dict(80,115,80,105, 0)
labels["00590"]=label_dict(205,190,130,85, 0)
labels["00626"]=label_dict(145,130,35,20, 1)
labels["04814"]=label_dict(100,127,30,50, 1)
# labels["04821"]=label_dict(-205,190,130,85)
labels["02362"]=label_dict(170,55,35,25, 0)
labels["02404"]=label_dict(145,275,50,85, 1)
labels["02440"]=label_dict(160,155,-15,10, 1)
labels["02444"]=label_dict(160,50,-420,330, 0)
img_index_dict ={}
idx = 0

for key in labels:
    img_index_dict[idx] = key
    idx+=1

def angle(label):
    width = 1080-label["left"]-label["right"]
    height = 1080-label["top"]-label["bottom"]
    x = math.acos(min(width,height)/max(width,height))
    degree = math.degrees(x) 
    return degree

def scale(label):
    width = 1080-label["left"]-label["right"]
    height = 1080-label["top"]-label["bottom"]
    #TODO consider rotation
    # return max(width,height)*1080/(label["size"]*540)
    return max(width,height)/label["size"]

def x_axis(label):
    return (label['left']-label['right'])*224/(2*label['size'])

def y_axis(label):
    return (label['top']-label['bottom'])*224/(2*label['size'])

def label_to_string(label):
    to_string = "angle:{},scale:{},x_axis:{},y_axis:{}".format(angle(label),scale(label),x_axis(label),y_axis(label))
    return to_string

def label_to_lst(label):
    return torch.tensor([angle(label),scale(label),x_axis(label),y_axis(label)])

if __name__ == "__main__":
    print(img_index_dict)
    # for key in img_index_dict:
    #     print("filename:{},".format(img_index_dict[key])+label_to_string(labels[img_index_dict[key]]))
    #     print(labels[img_index_dict[key]])