# NeuralCleanse Code
This is the section for NeuralCleanse. 

### Source Code

The first, OnnxConverter.py, consists of our testing converting .pt files to .onnx files. We experimented with attempting to convert whole folders (see below) before trying to solve the issues we were having on the reduced problem size of just converting one model. This failed and we discuss this in our Final Report.

The second file, convert_to_h5.py, was our large scale attempt to convert the MNIST-trained models from the TDC dataset (see ```Data``` folder from main root page of this repo) and convert them from .pt to .onnx to .pb file types. Unfortunately, while this actually succeeded, the .pb file type was incompatible with the version of TensorFlow necessitated by the NeuralCleanse library. Thus, while we were able to succeed in converting to .pb, we were unable to actually run any of the converted TDC models through NeuralCleanse. 

The remaining source code files in this folder are import requirements for the two listed above. The ```MetaNetwork.py``` file is just the PyTorch implementation of the networks required to convert the .pt file into a .onnx and ```utils.py``` and ```wrn.py``` are util functions and the implementation of WideResNet from the TDC source code required to fully load our models from .onnx to .pb filetypes. We don't make any modifications to these pieces of code but include them in case you may want to execute any of our conversion scripts yourself (see next section).

### Executable Code

We do not have any executable code or scripts to run for the NeuralCleanse approach in our project. Theoretically, if you modify the directory path in the convert_to_h5.py file to your downloaded copy of the TDC dataset, then you could run the script.


## Required Other Repositories
We made use of the official NeuralCleanse GitHub Repo: https://github.com/bolunwang/backdoor 
Make sure to download the repo and follow the instructions for setting it up.

We also used onnx-tensorflow: https://github.com/onnx/onnx-tensorflow 
We wanted to use this to help us convert our ONNX format models into TensorFlow but we obviously did not have good luck with this due to type/versioning conflicts between ONNX and the TensorFlow version of NeuralCleanse


Since most of the code for this method is from open-source repositories we list the opensource codebase statistics as mentioned in the Canvas assignment.


## Opensource codebase statistics
Backdoor: 1,159 lines, 91.1% Python, 8.9% Markdown

ONNX Tensorflow: 17,288 lines, 91.4% Python, 7.9% Markdown, 0.5% YAML, 0.1% Shell Scripts, 0% Properties (4 lines)

####

### Our Work/Contribution
The essence of our work on this task was running multiple trials to benchmark this method on both CPU and GPU devices. Thus, not a lot of code was necessary in performing these runs. Our codebase is limited to our attempts to convert our TDC dataset (see the ../Data directory) to TensorFlow. Results from those runs can be seen in our Final Report. We attempted to use the TDC dataset for this project but ended up just using different variations of the model linked here (https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing) by injecting it in different ways and with different triggers. This file is just a model trained on GTSRB. Performance was measured by taking the outputs of the NeuralCleanse ```gtsrb_injection_example.py``` and ```mad_outlier_detection.py``` scripts that can be found in the NeuralCleanse github linked above.



