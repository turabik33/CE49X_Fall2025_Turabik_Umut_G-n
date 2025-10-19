
---

# Python & Jupyter Setup Guide

This guide walks you through installing and configuring Python and Jupyter for our course.

---

## 1️⃣ Why Python & Jupyter?
- **Python**: A powerful, user-friendly language widely used in data science, engineering, and more.  
- **Jupyter**: An interactive environment (notebooks) for code, graphics, and explanatory text—all in one place.

---

## 2️⃣ Installing Python
We recommend the [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution for convenience.

### 🔹 Download & Install
1. **Download**  
   - Get Anaconda (full distribution) or Miniconda (lighter version).
2. **Install**  
   - Follow on-screen instructions for your OS (Windows, Mac, Linux).
3. **Verify Installation**  
   - Open a terminal (or Anaconda Prompt on Windows) and run:
     ```bash
     conda --version
     ```
   - You should see a version number (e.g., `conda 4.10.3`).

---

## 3️⃣ Creating a Conda Environment
To keep your work organized, create a dedicated environment for this course.

### 🔹 Steps:
1. **Create Environment**  
   ```bash
   conda create --name ce49x python=3.10
   ```
2. **Activate the Environment**  
   ```bash
   conda activate ce49x
   ```
3. **Install Required Packages**  
   ```bash
   conda install jupyterlab pandas numpy matplotlib seaborn scikit-learn
   ```

---

## 4️⃣ Running Jupyter
### 🔹 Start Jupyter
1. **Launch JupyterLab**  
   ```bash
   jupyter lab
   ```
   - This should open a new tab in your web browser.
2. **Create a Notebook**  
   - Click “+” (Launcher) → “Notebook” → select the `ce49x` kernel.

### 🔹 Test Your Setup
Run the following code in a new notebook:
```python
import sys
print("Hello, CE49X!")
print(f"Python version: {sys.version}")
```

If you see "Hello, CE49X!" and a Python version number, your setup is working correctly.

---

## 5️⃣ Troubleshooting
### 🔹 Common Issues & Fixes
- **Command not found**:  
  - Ensure Anaconda/Miniconda is installed properly.
  - Use the **Anaconda Prompt** (Windows) instead of cmd/PowerShell.
- **Wrong Python version**:  
  - Check your environments with:
    ```bash
    conda info --envs
    ```
  - Activate the correct one:
    ```bash
    conda activate ce49x
    ```
- **Package missing?**  
  - Try installing with:
    ```bash
    conda install <package_name>
    ```
  - Or if not available via Conda:
    ```bash
    pip install <package_name>
    ```

---

## 6️⃣ Best Practices
### 🔹 Environment Management  
✔️ Use separate environments for different projects.  
✔️ Keep your environment lightweight by only installing necessary packages.  

### 🔹 Frequent Updates  
Keep your tools up to date with:
```bash
conda update conda
conda update --all
```

### 🔹 Export & Import Environments  
For easy replication, export your environment:
```bash
conda env export > environment.yml
```
To recreate on another machine:
```bash
conda env create -f environment.yml
```

### 🔹 Organizing Notebooks  
✔️ Use markdown (`# Headings`, `**bold**`, `_italic_`) for notes.  
✔️ Label your code cells clearly.  
✔️ Keep notebooks modular—one topic per notebook.  

---

## 7️⃣ Additional Resources
- **Anaconda Documentation**: [https://docs.anaconda.com/](https://docs.anaconda.com/)  
- **Jupyter Documentation**: [https://jupyter.org/documentation](https://jupyter.org/documentation)  
- **Conda Cheat Sheet**: [Cheat Sheet PDF](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet/)  
- **Python for Data Science Handbook**: [Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/)  

---

📌 **If you run into any issues, open a GitHub Issue in this repository or reach out in the course discussion forum. Happy coding!** 🚀
