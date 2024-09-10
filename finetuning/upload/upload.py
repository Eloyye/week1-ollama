import os

from huggingface_hub import snapshot_download

from finetuning.upload.env import OLLAMA_CPP_PATH
from finetuning.upload.ollama_cl import export_to_ollama
from finetuning.upload.quantize import quantize_files


def main():
    model_name = "illusin/gemma-2-2b-chatdoctor"
    methods = ['q4_k_m']
    base_model_path = "./original_model/"
    quantized_path = "./quantized_model/"

    #
    snapshot_download(repo_id=model_name, local_dir=base_model_path, local_dir_use_symlinks=False)
    original_model = quantized_path+'/FP16.gguf'

    if not os.path.isdir(quantized_path):
        os.mkdir(quantized_path)

    os.system(f'python {OLLAMA_CPP_PATH}/convert-hf-to-gguf.py {base_model_path} --outtype f16 --outfile {original_model}')

    quantize_files(methods, quantized_path)

    # export_to_ollama('gemma-2-2b-chatdoctor')

if __name__ == '__main__':
    main()
