# EP2-mac0508

## Configuração no Google Colab

Comece pelo método mais simples (Google Colab).

### 1. Faça upload da pasta do projeto

Faça upload da pasta do projeto (EP2-mac0508) para o seu Google Drive.

### 2. Abra o notebook no Colab

File → Open notebook → Google Drive → EP2-mac0508/EP2.ipynb

### 3. Mude o runtime para GPU

Runtime → Change runtime type → Hardware accelerator: GPU

### 4. Monte o Google Drive

Execute em uma célula do notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 5. Ajuste a variável de caminho do projeto

Rode em célula:

```python
PROJECT_ROOT_DIR = "/content/drive/MyDrive/EP2-mac0508"
import os
os.chdir(PROJECT_ROOT_DIR)
```

### 6. Instale dependências

Execute em célula bash (coloque `!` no Colab):

```bash
!pip install transformers datasets evaluate sacrebleu pandas openpyxl torch peft sentencepiece scikit-learn matplotlib
```
