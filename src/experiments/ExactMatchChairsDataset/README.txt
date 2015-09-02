Exact-Match-Chairs Dataset
contains 315 images of 105 chair models.

---- directories ----
ExactMatchChairs
	contains all 315 cropped chair images.

raw_imgs.zip
	contains uncropped raw images

code
	code to prepare label files


---- files ----
(1) exact_match_chairs_img_filelist.txt
	image file names

(2) exact_match_chairs_cluttered_img_filelist.txt
	cluttered image file names (subset of above)

(3) exact_match_chairs_cluttered_img_indicies_0to314.txt
	index of image name of (2) in (1)

(4) filelist_chair_6777.txt
	all (shapenet core) chair model md5

(5) exact_match_chairs_img_modelIds_0to6776.txt
	chair model index of 315 images

(6) filelist_exactmatch_chair_105.txt
	105 (corresponding to the images) chair model md5

(7) exact_match_chairs_shape_modelIds_0to6776.txt
	index of 105 models of (6) in (4)

(8) exact_match_chairs_pool5_feat.npy
	pool5 features of 315 images
