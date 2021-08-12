import shutil, errno
import os
rootdir="/home/admin/桌面/数据集/caltech/JPEG/set00/V000"
new_path="/home/admin/桌面/caltech/train_3/images"
list=os.listdir(rootdir)
count=1
for filename in list:
	print(filename)
	crazystring = filename.split('_')[-1].split('.')[0]
        #new = crazystring[2:5]
	print(crazystring)
	new = crazystring[2:6]
	print(new)
	i = int(new)
	print(type(i))	
	if (i-2)%3==0:
		#src = os.path.join(rootdir, filename)
        	#dst = os.path.join(new_path, filename)
        	#print('src:', src)
        	#print('dst:', dst)
        	shutil.move(rootdir, new_path)	
	count+=1

















