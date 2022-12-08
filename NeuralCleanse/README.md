# NeuralCleanse Code
This is the section for NeuralCleanse. 

### Source Code
Not really any source code to speak of beyond OnnxConverter.py and convert_to_h5.py 

The first, OnnxConverter.py, consists of our testing converting .pt files to .onnx files. We experimented with attempting to convert whole folders (see below) before trying to solve the issues we were having on the reduced problem size of just converting one model. This failed and we discuss this in our Final Report.

The second file, convert_to_h5.py, was our large scale attempt to convert the MNIST-trained models from the TDC dataset (see ```Data``` folder from main root page of this repo) and convert them from .pt to .onnx to .pb file types. Unfortunately, while this actually succeeded, the .pb file type was incompatible with the version of TensorFlow necessitated by the NeuralCleanse library. Thus, while we were able to succeed in converting to .pb, we were unable to actually run any of the converted TDC models through NeuralCleanse. 


## Required Other Repositories
We made use of the official NeuralCleanse GitHub Repo: https://github.com/bolunwang/backdoor 
Make sure to download the repo and follow the instructions for setting it up.

We also used onnx-tensorflow: https://github.com/onnx/onnx-tensorflow 
We wanted to use this to help us convert our ONNX format models into TensorFlow but we obviously did not have good luck with this due to type/versioning conflicts between ONNX and the TensorFlow version of NeuralCleanse


Since most of the code for this method is from open-source we list the opensource codebase statistics as mentioned in the Canvas assignment


## Opensource codebase statistics
Backdoor: 1,159 lines, 91.1% Python, 8.9% Markdown

ONNX Tensorflow: 17,288 lines, 91.4% Python, 7.9% Markdown, 0.5% YAML, 0.1% Shell Scripts, 0% Properties (4 lines)

####

The essence of our work on this task was running multiple trials to benchmark this method on both CPU and GPU devices. Thus, not a lot of code was necessary in performing these runs. Our codebase is limited to our attempts to convert our TDC dataset (see the ../Data directory) to TensorFlow. Results from those runs can be seen in our Final Report.
