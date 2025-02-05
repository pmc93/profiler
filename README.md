# profiler

## 1. Download the Code from Git
First, clone the repository from GitHub (or another Git hosting service). Open a terminal or command prompt and run:

```bash
git clone https://github.com/pmc93/profiler.git
```

## 2. Navigate to the GUI Folder
Once the repository is downloaded, change the directory to the folder containing the GUI script:

```bash
cd path_to_repository/gui/version2
```

Replace `path_to_repository` with the name of the cloned repository.

## 3. Install Dependencies
The program requires external libraries, install them using:

```bash
pip install -r requirements.txt
```

If dependencies are managed using `conda`, create an environment and install them:

```bash
conda create --name my_env --file requirements.txt
conda activate my_env
```

## 4. Run the Python GUI Program
Once inside the `gui/version2` folder, execute the GUI script using:

```bash
python gui.py
```

If multiple Python versions are installed, you may need to use:

```bash
python3 gui_parent.py
```

## 5. Troubleshooting
- If the script does not run, ensure Python is installed by checking:
  
  ```bash
  python --version
  ```

- If dependencies are missing, install them as mentioned in Step 3.
- If you encounter permission issues, try running the command with `python -m`:
  
  ```bash
  python -m gui_parent.py
  ```

Profiler GUI program should now be running successfully!
