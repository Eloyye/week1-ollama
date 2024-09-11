# Ollama Week1

# About
This offers setup to explore text generation, sentiment analysis, summariation. It will also offer service to fine tune models using Google Colab.
Further it will handle deployment 

# Setup
1. First install dependencies
```shell
pip install -r requirements.txt
```
2. add environment variables in `./finetuning/env_.py` and change to `env.py`
3. Install llama.cpp
```shell
https://github.com/ggerganov/llama.cpp.git
```
4. install dependencies for importing model and quantizing model
```shell
pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

# Inference
1. To do inference there needs to be Q4_K_M.gguf file in `deployment/`
2. build image 
```shell
docker build -t <container_tag> . 
```
3. run image
```shell
docker run -p 8080:8080 -dit .
```

# Issues
1. More streamline deployment