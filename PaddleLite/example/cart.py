from paddlelite import *
import numpy as np
import time

import cv2
import time
import os

crop_size = 128;

def preprocess(img):
	img = cv2.resize(img, (int(crop_size), int(crop_size)))
	img = np.array(img).astype(np.float32)
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	img = img / 255.0;
	return img;

def load_model():
	valid_places = (
		Place(TargetType.kFPGA, PrecisionType.kFP16, DataLayoutType.kNHWC),
		Place(TargetType.kHost, PrecisionType.kFloat),
		Place(TargetType.kARM, PrecisionType.kFloat),
	);
	config = CxxConfig();
	model = "cart";
	model_dir = model;
	config.set_model_file(model_dir + "/model")
	config.set_param_file(model_dir + "/params")
	config.set_valid_places(valid_places);
	predictor = CreatePaddlePredictor(config);
	return predictor;

def predict(predictor, image_path, z):
	print(image_path)
	src = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
	img = preprocess(src);

	i = predictor.get_input(0);
	i.resize((1, 3, 128, 128));

	z[0, 0:img.shape[0], 0:img.shape[1] + 0, 0:img.shape[2]] = img
	z = z.reshape(1, 3, 128, 128);
	i.set_data(z)

	predictor.run();
	out = predictor.get_output(0);
	score = out.data()[0][0];
	print(out.data()[0])
	return score;

def iterate(directory):
	score_map = {};
	file1 = open("score_map.txt","w+") 

	for filename in os.listdir(directory):
		print(filename)
		path = "{}/{}".format(directory, filename);
		score = predict(predictor, path, z)
		file1.write(filename);
		file1.write("  ");
		file1.write(str(score));
		file1.write("\n");

	file1.close();

z = np.zeros((1, 128, 128, 3))
predictor = load_model();

predict(predictor, "images/8081.jpg", z)
print(predict(predictor, "images/8081.jpg", z))
# iterate("hsv_img")