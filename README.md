

Steps to install all requirements:
1. install all modules found in requirements.txt file.
2. for cocoapi tools there are some changes to be made depending on the os
    - Linux: same as on repo (https://github.com/cocodataset/cocoapi.git)
    - Win: setup.py file has to be changed, se replace:
        - line: extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
	- with line: extra_compile_args={'gcc': ['/Qstd=c99']},
    - make sure there are installed Cython (should be included in requirements file)
    and C++ 14.0 (or higher) build tools => this last requirement is applicable only
    for Win machines, no need on linux based.

Observations: if there are some errors while isntalling PythonAPI, try to set the
    right path, one of the following: cocoapi/ or cocoapi/PythonAPI
