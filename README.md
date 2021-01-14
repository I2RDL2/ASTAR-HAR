# A*HAR: A NEW BENCHMARK TOWARDS SEMI-SUPERVISED LEARNING FOR CLASS IMBALANCED HUMAN ACTIVITY RECOGNITION

The tensorflow code for "A\*HAR: A NEW BENCHMARK TOWARDS SEMI-SUPERVISED LEARNING FOR CLASS
IMBALANCED HUMAN ACTIVITY RECOGNITION " (https://arxiv.org/abs/2101.04859)


The environment can be found in dockerhub:
docker pull loklu/mt_tensorflow:tf1.2.1_py35_lib2

 The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
pip install tensorflow==1.2.1 numpy scipy pandas
./prepare_data.sh
``` 

To train the model, run:

* 'python /experiments/har/har.py --template X' for HAR models.
 modify arguments in template.py for choosing type of experiment and no: of unlabeled samples , Where X can be defined as suitable template

