DIRBAK=$(pwd)
pip install -r requirements.txt

pip install vllm==0.14.1
pip install ../data/wheels/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install uv==0.10.0


pip install english-to-ipa@git+https://github.com/mphilli/English-to-IPA

#sudo apt install -y sox libsox-fmt-all
