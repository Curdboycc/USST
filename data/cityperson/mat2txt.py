#coding=utf-8
import scipy.io
import os
 
#Function：将训练集的annotations转换为YOLOv3训练所需的label/train/XXX.txt格式
#How to run? ###python citypersons2yolo.py
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
 
#You can download anno_train.mat from "https://bitbucket.org/shanshanzhang/citypersons/src/f44d4e585d51d0c3fd7992c8fb913515b26d4b5a/annotations/".   
data = scipy.io.loadmat('annotations/anno_train.mat')
data = data['anno_train_aligned'][0]
 
if not os.path.exists('labels/train/'):
	os.makedirs('labels/train/')
 
 
for record in data:
	im_name = record['im_name'][0][0][0]
	bboxes = record['bbs'][0][0]
	(shot_name, extension) = os.path.splitext(im_name)
	txt_name = os.path.join('labels/train', shot_name+'.txt')
	f = open(txt_name, 'w')
	#im_name = os.path.join('train', im_name.split('_', 1)[0], im_name)
 
	for bbox in bboxes:
		class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis = bbox
		if class_label == 0:
			continue
		b = (float(x1), float(x1+w), float(y1), float(y1+h)) #(xmin, xmax, ymin, ymax)
		bb = convert((int(2048), int(1024)), b)
		f.write('0 ' + ' '.join([str(a) for a in bb]) + '\n')
 
	f.close()

