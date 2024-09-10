import ollama


def export_to_ollama(model_name):
    with open('./import_to_ollama/Modelfile', 'r') as file:
        modelfile = file.read()
        response = ollama.create(model=model_name, modelfile=modelfile)
        print(response)

if __name__ == '__main__':
    export_to_ollama('gemma-2-2b-chatdoctor')