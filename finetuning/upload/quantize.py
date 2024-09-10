import os

from finetuning.upload.env import OLLAMA_CPP_PATH

def quantize_files(methods, quantized_path):
  for m in methods:
    qtype = f"{quantized_path}/{m.upper()}.gguf"
    os.system(f"{OLLAMA_CPP_PATH}/llama-quantize "+quantized_path+"/FP16.gguf "+qtype+" "+m)
