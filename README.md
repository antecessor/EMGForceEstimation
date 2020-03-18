# EMGForceEstimation
## Installation
1. Clone the code
2. create a virtual environment or use pycharm to do it for you
3. pip install -r requirements.txt

## Force estimation pre-trained model
1. open test/test_plotForcePredicted
2. correct directories:
    - getSignal3(...,path="your data path folder") ; see inside function for more details.
    - model = tf.keras.models.load_model(YourmodelPathFile.h5)
3. run test_plotForcePredicted

## Force estimation training
1. DeepLeraning/lstm.py
2. correct directories:
    - getSignal3(...,path="your data path folder") ; see inside function for more details.
3. Run

## System Requirements    
- You can use CPU or tensorflow-gpu==1.15 
- I personally used Adapter Type	GeForce GTX 1060 6GB, NVIDIA compatible
