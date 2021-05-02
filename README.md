# Introduction

## Where do I start?
Here. This project involves multiple repositories, each with their own README and instructions. Start by reading this document, and we will direct you to the other projects and their READMEs as we go along.

## Do I need to read the report?
Reading the final report is necessary to understanding the reasoning behind the technical decisions made, which will probably make it easier to navigate this project and make further enhancements. It is also crucial to read Section 4 of the report to understand how to emulate low precision types in the simulator.

However, reading this README and following all the steps should be sufficient for you to set up this project and start running programs on the simulator.

## Where are all the source files?
The source repositories are located in the [`variable-precision-gpu` GitHub organization](https://github.com/variable-precision-gpu), which I link to throughout this document. However, a clone of each repository should also be present in this ZIP file.

# [GPGPU-Sim](https://github.com/variable-precision-gpu/gpgpu-sim/)
This is the modified version of the original GPGPU-Sim, forked from the [GitHub repository](https://github.com/gpgpu-sim/gpgpu-sim_distribution). The enhancements that we have added allow the simulator to execute operations in low precision, thereby achieving mixed precision deep learning.


## Prerequisites
- Ubuntu, GCC: Ensure that system is on Ubuntu 16.04/18.04, and gcc is set to gcc 4/5. Other versions are not guaranteed to work.

- CUDA: Install the CUDA 8 Toolkit [(if using closed-source libraries like cuBLAS or cuDNN)](https://github.com/gpgpu-sim/gpgpu-sim_distribution/issues/166#issuecomment-604505230) or otherwise any compatible version of CUDA listed on the simulator README (e.g. 9.0, 9.1, 10, and 11). Among the existing projects, MLP and CNN use cuBLAS and will require CUDA <=8.

- Others: Install all the dependencies in the simulator [README (Step 1)](https://github.com/variable-precision-gpu/gpgpu-sim#step-1-dependencies) using the provided commands. Note that the library `libglut3-dev` is no longer available and should be replaced with `freeglut3-dev`.

- MPFR: An additional dependency introduced in our modified simulator, you will need to install MPFR 3.1 in order to use variable precision types.

Follow the instructions in the simulator README to install cuDNN if you plan to add cuDNN projects - none of the existing projects (MLP, MLP2, CNN, CNN2) do.

## Usage
### 1. Building the simulator binary
This step remains unchanged from the original simulator. Instructions are in the simulator [README (Step 2)](https://github.com/variable-precision-gpu/gpgpu-sim#step-2-build).

### 2. Running a CUDA program on the simulator
You should be able to use this version of the simulator in the same manner as the original, if you wish to execute programs without varying precision [README (Step 3)](https://github.com/variable-precision-gpu/gpgpu-sim#step-3-run).

You will need to familiarize yourself with the execution modes of the simulator and the general workflow in order to proceed.

For how to run programs on the simulator with variable precision, refer to the workflow presented in the following section.

# [GPGPU-Sim Deep Learning Runner (sim-dl-runner)](https://github.com/variable-precision-gpu/sim-dl-runner/)
---
This section is identical to the README in the `sim-dl-runner` repository. If you've read one, feel free to ignore the other.

---

This project serves to organize the files required for running mixed-precision deep learning on the simulator. In addition, the `runner.py` script configures and serves as the entry point for execution runs, automating the various steps involved.

```
├─ config/
│  ├─ config_volta_islip.icnt
│  ├─ gpgpusim.config
├─ programs/
│  ├─ CNN2/
│  ├─ MLP2/
├─ README.md
├─ runner.py
├─ utils.py
```

`config/`: Central location containing the simulator configuration files to be applied to all programs. Replace with other config files in the simulator repository to simulate other GPUs.

`programs/`: Directory for deep learning projects as git submodules. Making the projects submodules allows the repositories to be updated independently.

`runner.py`: Python 3 script automating the simulator set up and clean up. Refer to the inline documentation in the script for details about the configuration parameters.

`utils.py`: Python 3 utility functions. Of note, `mpfr_exponent_range()` computes the environment variables one should set for the VF32 type in order to emulate a specific exponent and significand width.

## Prerequisites
- Python:  >=3.7
- Modified GPGPU-Sim

## Usage

### 1. Configuring the simulator
Place the configuration files of the GPU being simulated in the `configs/` folder, instead of directly in the deep learning program's directory.

The mixed precision experiments conducted so far were performed with the TITAN V configuration that's already in the folder, but feel free to experiment with other configurations. Note that some runtime issues were encountered previously with the TITAN X config.

### 2. Adding new deep learning programs
Add a new deep learning application into the programs subdirectory as a git submodule

```console
$ git submodule add <repo_url>
```

All new programs added will need to be made compatible with the runner script. This can be done by modifying the program to conform to the following rules:

1\. Program should compile with `make`, without any parameters. This means the first rule in the Makefile should be the build command.

2\. The program will have to be modified to split the training and inference stages, saving the trained weights to a specified file. This will facilitate the independent execution of each of these stages, crucial given the long execution time of deep learning programs on the simulator. The main function should be modified to accept the following "modes":

#### Train
```console
$ <program> -train <epochs> <weights_file>
```
- `<program>` the program binary
- `-train` specifies that training should be executed
- `<epochs>` number of epochs to train for
- `<weights_file>` the file to save the model's trained weights to


#### Train (Increment)
```console
$ <program> -train-increment <start_epoch> <end_epoch> <input_weights_file> <output_weights_file>
```
- `<program>` the program binary
- `-train-increment` specifies the incremental training mode
- `<start_epoch>` number of epochs that have already been trained
- `<end_epoch>` the epoch number to terminate training
- `<input_weights_file>` the file that contains the weights trained prior to the point of program execution
- `<output_weights_file>` the file to save the weights post this round of training

#### Test
```console
$ <program> -test <weights_file>
```
- `<program>` the program binary
- `-test` specifies that inference should be executed
- `<weights_file>` the file containing the trained weights

Each of the programs in the [`variable-precision-gpu` GitHub organization](https://github.com/variable-precision-gpu) have already been made compatible with the above rules and can serve as examples.

3\. Generate and override PTX accordingly, if using mixed precision

### 3. Runner execution
Before executing the runner script `runner.py`, configure the execution parameters first. A brief description of what the script does is provided at the top of the file for reference. Specify the target program, the name of the executable and the stages to run.

You can then run the script with:
```console
$ python3 runner.py
```

### 4. Overriding PTX
In order to perform mixed precision deep learning, you first have to run the target program once on the simulator to generate PTX files.

Then, go to the `configs/gpgpusim.config` and set the `-gpgpu_generate_ptx` option to `0` - this turns off PTX generation, and the next time the simulator is run, it will ingest PTX files directly.

Then, edit the PTX files to override PTX instructions, setting the types of the instructions to `VF32`. Instructions marked as `VF32` will be performed at an arbitrary precision, depending on the values of the `VF_SIGNIFICAND`, `VF_EXPONENT_MIN` and `VF_EXPONENT_MAX` environment variables at the time of kernel invocation.

For details on how exactly to perform PTX overriding, including what parameters to set to emulate certain precisions, please refer to Section 4 of the final report.

While you can configure the aforementioned `VF32` environment variables in the `runner.py` script to apply to an entire execution run, if you would like to change `VF32` precision within a run (e.g. between neural network layers), you will have to set the environment variables in the application code itself. For example:
```c++
// set VF32 precision and range to be that of bfloat16
setenv("VF_SIGNIFICAND", "8", 1);
setenv("VF_EXPONENT_MIN", "-132", 1);
setenv("VF_EXPONENT_MAX", "128", 1);
```


# Deep Learning Programs
The following programs have been experimented with on the simulator as part of our project. Limited work was performed on the `MLP` and `CNN` models, since both of them used cuBLAS, which make PTX overriding challenging (see Section 5.1 of the report).

### [MLP](https://github.com/variable-precision-gpu/MLP)
- Uses cuBLAS
- MNIST

### [CNN](https://github.com/variable-precision-gpu/CNN)
- Uses cuBLAS
- MNIST

### [MLP2](https://github.com/variable-precision-gpu/MLP2)
- No external dependencies
- MNIST

### [CNN2](https://github.com/variable-precision-gpu/CNN2)
- No external dependencies
- MNIST and CIFAR

Please use the above programs as references on how to make CUDA programs compatible with the simulator, as well as how PTX overriding and VF32 precision/range setting are done. The git commit history might be useful in highlighting the key changes.

# Contact
If you run into any issues, need any clarification or would like to find out more, you can contact me at [my email address](mailto:au.liangjun@gmail.com).