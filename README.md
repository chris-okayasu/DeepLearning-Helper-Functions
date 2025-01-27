# Deep Learning Helper Functions

This repository contains a set of helper functions for TensorFlow, designed to streamline the development of artificial intelligence projects.

## Installation in Google Colab / Jupyter Notebook

To install and use this package in Google Colab, follow these steps:

1. **Clone the repository:**

    In a Google Colab cell, run the following command to clone the repository:

   ```bash
   !git clone https://github.com/chris-okayasu/DeepLearning-Helper-Functions.git
   ```

2. **Install the package:**
   
   After cloning the repository, navigate to the project directory and install the package locally:
   
   ```bash
   %cd DeepLearning-Helper-Functions
   !pip install .
   ```
## Example Usage

Once you have installed the package, you can import and use the functions as shown in the following example:

```py
    from ai_helpers import fun_nvidia
    fun_nvidia()
```

> **Note**🔑: If you want to import or install a single function (and code) you can also ->

```py
    !wget https://raw.githubusercontent.com/chris-okayasu/DeepLearning-Helper-Functions/refs/heads/main/ai_helpers/visualize/fun_nvidia.py
```
This ⬆️ are going to create a file .py with the name of the function and inject the code in your project