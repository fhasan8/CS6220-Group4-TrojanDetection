# NeuralCleanse Code
This is the section for NeuralCleanse. 

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
