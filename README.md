# Semi-supervised Audio Classification with Consistency-Based Regularization

This is the TensorFlow source code for the "Semi-supervised Audio Classification with Consistency-Based Regularization" paper.

The environment can be found in dockerhub:
docker pull loklu/mt_tensorflow:tf1.2.1_py35_lib2

 The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
pip install tensorflow==1.2.1 numpy scipy pandas
./prepare_data.sh
``` 

To train the model, run:

* `python train_svhn.py` to train on SVHN using 500 labels
* `python train_cifar10.py` to train on CIFAR-10 using 4000 labels