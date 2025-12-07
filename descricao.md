# EP2 — Tradução Português ↔ Tupi Antigo

Este projeto implementa a tarefa proposta no EP2 MAC0508
utilizando modelos transformer multilíngues, com treinamento
e avaliação nos sentidos PT→TA e TA→PT.

## Corpus data.xlsx
O arquivo `data.xlsx` localizado na raiz do projeto contém
frases paralelas Português ↔ Tupi Antigo. Os nomes das colunas
são **exatamente**:

- Português
- Tupi Antigo

Preservamos acentos, diacríticos e grafia histórica.
Normalizações excessivas podem remover significado linguístico,
portanto realizamos apenas limpeza mínima (espaços, caracteres
invisíveis).

## Pipeline
- leitura e limpeza
- split (70/15/15)
- zero-shot
- fine-tuning
- avaliação BLEU, chrF1, chrF3
- comparação
- exemplos qualitativos

## Modelos
O modelo base utilizado é `facebook/mbart-large-50-many-to-many-mmt`.
Como o mBART não possui idioma nativo para Tupi Antigo, utilizamos
prefixos explícitos: `<pt>` e `<tupi>` nas entradas e saídas do modelo.

## Execução
Abra o notebook `EP2.ipynb` no Jupyter Lab ou Colab.
O notebook instalará dependências e executará todos os passos.

## Resultados
Os resultados são gravados em:
- `results/results_zero_shot.json`
- `results/results_few_shot.json`
- `results/outputs_zero_shot/`
- `results/outputs_few_shot/`

## Métricas
Conforme enunciado do EP2:
- BLEU
- chrF1
- chrF3

O notebook contém explicações matemáticas das métricas.

## Estrutura
data.xlsx
EP2.ipynb
MBart50.ipynb
data/
models/
results/

## Observações linguísticas
O Tupi Antigo possui variações históricas, símbolos especiais e
grafia não moderna. Não realizamos normalização agressiva.
Explicamos isto no relatório dentro do notebook.

## Licença
Uso acadêmico para fins de EP2 MAC0508.
