# PANet


## Model
![PANet Architecture](images/Framework.png)



## Dataset
![PANet Dataset](images/dataset_samples.png)


spiral, circles, moons, and blobs consist of the dataset.

Train : Valid : Test = 8 : 1 : 1

X_train.npy (800, 500, 2), X_valid.npy (100, 500, 2), X_test.npy (100, 500, 2): 2D point distributions

y_train.npy (800, 500, 6), y_valid.npy (100, 500, 6), y_test.npy (100, 500, 6): the number of clusters / one-hot encoding

y_train_clustering.npy (800, 500, 2), y_valid_clustering.npy (100, 500, 2), y_test_clustering.npy (100, 500, 2): clustering label for each point
