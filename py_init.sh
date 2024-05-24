# Linux
chmod 700 py_init.sh
conda env create -f environment.yml
conda activate traderl
mkdir -p config source test utils && touch main.py requirements.txt config/__init__.py source/__init__.py test/__init__.py utils/__init__.py