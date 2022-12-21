import os
import re
import numpy as np

def decompose_path(lines: str, root_fashion_dir: str):

	# 1. Decompose the file name
	lines = lines[7:] # Remove the fashion vocabulary.
	FLAG = True if 'WOMEN' in lines else False
	
	if FLAG: 
		result = re.search(r'(WOMEN)(.+)(id\d{8})(.+\.jpg)', lines)
		file_name = [result.group(i) for i in range(1, 5)]
		file_name[2] = re.sub('id', 'id_', file_name[2])
		file_name3_temp = re.search(r'\d{2}_\d+', file_name[3])
		file_name3_replace = file_name3_temp.group(0) + "_"
		file_name[3] = re.sub(file_name3_temp.group(0), file_name3_replace, file_name[3])
	else:
		result = re.search(r'(MEN)(.+)(id\d{8})(.+\.jpg)', lines)
		file_name = [result.group(i) for i in range(1, 5)] # List[str]
		file_name[2] = re.sub('id', 'id_', file_name[2])
		file_name3_temp = re.search(r'\d{2}_\d+', file_name[3])
		file_name3_replace = file_name3_temp.group(0) + "_"
		file_name[3] = re.sub(file_name3_temp.group(0), file_name3_replace, file_name[3])
	
	# 2. Compose the file path
	root_dir = os.path.join(root_fashion_dir, 'img')
	for item in file_name:
		root_dir = os.path.join(root_dir, item)
	
	return root_dir


def check_files_path(root_fashion_dir: str, choice: str):
	"""
	choice: 'train' or 'test'
	"""

	train_images = []
	root_fashion_dir_deep = os.path.join(root_fashion_dir, 'deepfashion')
	file_lst = 'train.lst' if choice == 'train' else 'test.lst'
	with open(os.path.join(root_fashion_dir_deep, file_lst), 'r') as train_f:
		for lines in train_f:
			lines = lines.strip()
			if lines.endswith('.jpg'):
				train_images.append(lines)
	
	print('The length of train_images is {0}'.format(len(train_images)))

	train_is_file_results = [os.path.isfile(decompose_path(line, root_fashion_dir)) for line in train_images]
	train_is_file_results = np.array(train_is_file_results)
	train_is_file_paths = [decompose_path(line, root_fashion_dir) for line in train_images]

	print('All files are found : {0}'.format(np.all(train_is_file_results)))

	return train_is_file_paths, root_fashion_dir_deep, train_images


def transfer_file(root_fashion_dir: str, choice: str):

	is_file_paths, root_deepf, images_names = check_files_path(root_fashion_dir, choice)

	# 1. Establish a new file path
	root_deepf = os.path.join(root_deepf, 'fashion_resize')
	train_or_test_path = os.path.join(root_deepf, choice)
	if not os.path.exists(train_or_test_path):
		os.makedirs(train_or_test_path)
	
	# 2. Copy the file
	for from_, img_name in zip(is_file_paths, images_names):

		to_ = os.path.join(train_or_test_path, img_name)
		os.system('cp %s %s' %(from_, to_))

# path for downloaded fashion images
# root_fashion_dir = 'your_path/deepfashion'

root_fashion_dir = '/media/alien/c44d249a-e622-42e6-bcb5-9bf635999267/Controllable_Person_Image_Synthesis_with_Attribute_Decomposed_GAN'
assert len(root_fashion_dir) > 0, 'please give the path of raw deep fashion dataset!'

transfer_file(root_fashion_dir, choice= 'train')

"""
root_dir = decompose_path('fashionMENTees_Tanksid0000122204_3back.jpg', root_fashion_dir)
print(root_dir)
print(os.path.isfile(root_dir))
"""
