# Getting the Codebase Setup

## Create a Virtual Environment
First, create the virtual environment with:
```bash
python -m venv .venv
```

Then activate the venv.
```bash
# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\Activate
```

## Install dependencies using `requirements.txt`
Install the `requirements.txt` in venv with:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If adding or changing packages, create an updated `requirements.txt` before committing, using:
```bash
pip freeze > requirements.txt
```
