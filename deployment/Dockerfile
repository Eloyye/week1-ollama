FROM ollama/ollama
LABEL authors="eloyyeluo"

WORKDIR /app

COPY . /app

ENV OLLAMA_HOST=0.0.0.0:8080

ENV OLLAMA_MODELS=/models

ENV OLLAMA_DEBUG=false

ENV OLLAMA_KEEP_ALIVE=-1

ENV MODEL=medical-bot

RUN ollama serve & sleep 5 && ollama create $MODEL -f ./Modelfile

ENTRYPOINT ["ollama", "serve"]