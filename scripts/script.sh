#sample script will be updated 

cd "$(dirname "$0")/.."

pip install -r requirements.txt && cd data && python3 download_dataset.py && python3 ../scripts/script.py