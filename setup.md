# Setup

## Step 1: Create and activate virtual environment

```bash
python -m venv rl.venv
source rl.venv/bin/activate
pip install --upgrade pip
```

## Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Install PyTorch for ROCm (AMD GPU)

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

## Step 4: Install Atari ROMs

```bash
AutoROM --accept-license --install-dir "$(python -c 'import ale_py.roms; print(ale_py.roms.__file__.replace("__init__.py", ""))')"
```
