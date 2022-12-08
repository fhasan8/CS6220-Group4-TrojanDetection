# STRIP Code
This is the section for the STRIP model. 

## Source Code
We made use of the official STRIP GitHub Repo: https://github.com/garrisongys/STRIP 

We also used onnx-tensorflow: https://github.com/onnx/onnx-tensorflow 
We used this to help convert our ONNX format models into TensorFlow but we did not have progress with this due to type/versioning conflicts between ONNX and the TensorFlow version of STRIP.

####

Our work on this model was taking the TensorFlow implementation of STRIP provided in the official Github Repo and creating our own PyTorch implementation of the model algorithm. We manually ran our PyTorch implementation of STRIP on 6 TDC models. 

We initially attempted to make use of the original TensorFlow implementation by converting our TDC dataset (see the ../Data directory) to TensorFlow, but faced version conflicts during the conversion process as mentioned above.
