# BaTCAVe


## Structure
This project is organized into the following directories:
- `utils/`: Contains the important library code to be replaced by the user.
- `Exp 1/`: Contains the code for the first experiment.
- `Exp 2/`: Contains the code for the second experiment.
- `Exp 3/`: Contains the code for the third experiment.
- `Exp 4/`: Contains the code for the fourth experiment.

## Getting Started
These instructions will help you use BaTCAVe on your local machine.

### Install with pip
```bash
pip install -r requirements.txt
```

### Updating the library code
The library code to be replaced by the user is located in the `utils/` directory. The user should replace the code in the `utils/` directory as explained below.
- `utils/tcav.py`: Replace the `tcav.py` in your `<your_venv>/lib/python/site-packages/captum/concept/_core` folder with this file.
- `utils/__init__.py` : Replace the `__init__.py` in your `<your_venv>/lib/python/site-packages/captum/concept` folder with this file.


## Note
Since the pretrained models were too large to share in supplementary we have skipped those for now, but will be making them available later in private repo.