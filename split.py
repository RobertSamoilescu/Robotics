import numpy as np
import os

if __name__ == "__main__":
	train_directory = "./dataset"
	train_dataset = os.listdir(train_directory)
	
	# create test set
	no_test = int(len(train_dataset) * 0.01)
	test_dataset = list(np.random.choice(train_dataset, no_test, replace=False))
	
	# move files
	test_directory = "./test_dataset"
	try:
		os.mkdir(test_directory)
	except:
		print("already exists or something wrong")

	for file in test_dataset:
		source = os.path.join(train_directory, file)
		destination = os.path.join(test_directory, file)
		os.rename(source, destination)