import json
import os
import re

gpu_ids = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

root_path = '/home/admin/桌面/off'
list = os.listdir(root_path)
count = 1
for i in range(0,len(list)):
	
	n = str(i+6)
	s = n.zfill(3)
	

	path = os.path.join(root_path,s)
	#print(path)

	f = open(os.path.join(path,'val_det.txt'),'r')
	info = []
	
	for lines in f:
		
		#print(type(lines))
		parts = lines.strip(' ').split()
		
		parts[0] = lines.split(' ')[0]
		parts[1] = lines.split(' ')[1]

		parts[2] = lines.split(' ')[2]
		parts[3] = lines.split(' ')[3]
		parts[4] = lines.split(' ')[4]
		parts[5] = lines.split(' ')[5]
		id = count		
		image_id = float(parts[0])
		category_id = 1
		x1 = float(parts[1])
		y1 = float(parts[2])
		x2 = float(parts[3])
		y2 = float(parts[4])
		score = float(parts[5])
		#print(x1)

		
		info.append({'id':id,'image_id': image_id,'category_id': category_id,'bbox': [x1, y1, x2, y2],'score':score})
		print(info)
		
		count+=1

	
	path = '/home/admin/桌面/off/'
	folder = os.path.join(path,s)
	if not os.path.exists(folder):
  		os.makedirs(folder)
	phase = 'val_det'
	json_name = os.path.join(folder,'{}.json'.format(phase))
	with open(json_name, 'w') as f:
  		json.dump(info, f)
	
