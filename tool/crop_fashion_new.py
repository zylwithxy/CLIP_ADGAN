from PIL import Image
import os
from tqdm import tqdm

def list_file(img_dir: str, choice: str):
	"""
	img_dir: The root of the dataset.
	choice: train or test.
	"""
	img_deepf = os.path.join(img_dir, 'deepfashion')

	img_deepf_resize_before = os.path.join(img_deepf, 'fashion_before_resize')
	img_deepf_resize_after = os.path.join(img_deepf, 'fashion_resize')

	img_deepf_chosen = os.path.join(img_deepf_resize_before, choice)
	img_deepf_resize_after_chosen = os.path.join(img_deepf_resize_after, choice)

	# Create directories
	if not os.path.exists(img_deepf_resize_after_chosen):
		os.makedirs(img_deepf_resize_after_chosen)

	list_files = os.listdir(img_deepf_chosen)
	print(f"The number of current files is {len(list_files)}")
	
	# Crop each image in list_files
	for filename in tqdm(list_files):
		if not filename.endswith('.jpg') and not filename.endswith('.png'):
			continue

		img = Image.open(os.path.join(img_deepf_chosen, filename))
		imgcrop = img.crop((40, 0, 216, 256)) # Center crop
		imgcrop.save(os.path.join(img_deepf_resize_after_chosen, filename))
	 

img_dir = '/media/alien/c44d249a-e622-42e6-bcb5-9bf635999267/Controllable_Person_Image_Synthesis_with_Attribute_Decomposed_GAN'

list_file(img_dir, choice= 'test')

"""

if not os.path.exists(save_dir):
	os.mkdir(save_dir)

cnt = 0

for item in os.listdir(img_dir):
	if not item.endswith('.jpg') and not item.endswith('.png'):
		continue
	cnt = cnt + 1
	print('%d/8570 ...' %(cnt))
	img = Image.open(os.path.join(img_dir, item))
	imgcrop = img.crop((704, 0, 880, 256))
	imgcrop.save(os.path.join(save_dir, item))

"""
