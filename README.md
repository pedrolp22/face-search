# Buscador de rostos
Projeto simples de sistema para busca de rostos com base em um índice de vetores pré-calculados.

# Como executar

* Clone o repositório:
```bash
git clone https://github.com/seuusuario/face-search-engine.git
cd face-search-engine
```
* Crie um venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
* Atualize as ferramentas de empacotamento:
```bash
pip install --upgrade pip setuptools wheel
```
* Instale as bibliotecas
```bash
pip install face_recognition opencv-python numpy faiss-cpu fastapi uvicorn
```
* Defina o dataset (para testes, foi utilizado o dataset **[Labeled Faces in the Wild](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?resource=download)**)

Coloque os arquivos do seu dataset na pasta dataset/nome-do-dataset e altere o arquivo generate_embeddings.py para refletir isso (usa o LFW por padrão)

* Gere os embeddings com o script: 
```bash
python scripts/generate_embeddings.py
```
Arquivos gerados:
```
embeddings/
   embeddings.npy
   metadata.json
```

* Execute o script para construir o índice
```bash
python scripts/build_index.py
```
Arquivos gerados:
```
embeddings/faiss.index
```
* Execute o script search_face.py com o caminho da imagem como parâmetro:

Exemplo:
```bash
python scripts/search_face.py test_images/query.jpg
```
O script retorna os 5 rostos mais semelhantes do dataset (menor distância entre os vetores).

