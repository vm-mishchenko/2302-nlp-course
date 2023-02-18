- `faiss-gpu` package presumably requires
  - Python :: 2.7
  - Python :: 3.5
  - Python :: 3.6
  - Python :: 3.7
- mac m1 does not support easily these Python versions
- used `https://www.python.org/downloads/macos/` to install Python 3.7
- downloaded `macOS 64-bit installer` and just clicked next, next, next
- got some another error
- replaced `faiss-gpu` to `faiss-cpu`
- `faiss-cpu` was installed without problem for  Python 3.7
- did not try for other Python environments
- not sure what the diff between `faiss-gpu` and `faiss-cpu`, but everything inside `NLP_Week_1_Word2Vec_Tutorial.ipynb` works

- `faiss-gpu` vs `faiss-cpu`
  - `faiss-cpu` is a lightweight version of the library that only supports CPU hardware. It does not require any additional dependencies beyond what is already installed on a typical system.

  - `faiss-gpu` is an extension of the main library that provides GPU support. It requires installation of additional dependencies, such as CUDA and cuDNN, and is typically used for large-scale similarity search problems that require high performance on specialized hardware

