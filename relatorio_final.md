# Relatório - EP2: Tradução Automática de Baixo Recurso
## Português ↔ Tupi Antigo

---

**Disciplina:** MAC0508 — Introdução ao Processamento de Língua Natural  
**Autor:** Gabriel Ferreira de Souza Araujo  
**NUSP:** 12718100  
**Data de Entrega:** 07/12/2025

---

## Sumário

1. [Introdução](#1-introdução)
2. [Desenvolvimento do Projeto](#2-desenvolvimento-do-projeto)
3. [Justificativa de Escolha de Modelos e Hiperparâmetros](#3-justificativa-de-escolha-de-modelos-e-hiperparâmetros)
4. [Resultados](#4-resultados)
5. [Desafios Encontrados](#5-desafios-encontrados)
6. [Possíveis Melhorias](#6-possíveis-melhorias)
7. [Como Rodar o Projeto](#7-como-rodar-o-projeto)
8. [Conclusão](#8-conclusão)
9. [Referências](#9-referências)

---

## 1. Introdução

Este relatório documenta o desenvolvimento, implementação e avaliação de modelos de tradução automática para o par linguístico **Português ↔ Tupi Antigo**, uma tarefa desafiadora devido à escassez de dados disponíveis para esta língua histórica brasileira.

### 1.1 Objetivo do Projeto

O objetivo principal foi implementar e comparar tradutores automáticos em dois regimes distintos:

1. **Zero-shot Learning**: Utilização de modelos pré-treinados sem qualquer ajuste fino no corpus Português-Tupi Antigo
2. **Few-shot Learning (Fine-tuning)**: Ajuste fino de modelos utilizando o corpus paralelo disponível

A comparação foi realizada em ambas as direções de tradução:
- **Português → Tupi Antigo (PT → TA)**
- **Tupi Antigo → Português (TA → PT)**

### 1.2 Modelos Utilizados

#### Zero-shot: NLLB-200 (facebook/nllb-200-distilled-600M)

Para o regime zero-shot, utilizei o modelo **NLLB-200** (No Language Left Behind) da Meta AI. A escolha deste modelo foi estratégica: embora não possua suporte nativo ao Tupi Antigo, ele inclui o **Guarani** (`grn_Latn`), uma língua da família Tupi-Guarani geneticamente relacionada ao Tupi Antigo. Esta proximidade linguística permite uma aproximação razoável via transferência de conhecimento entre línguas aparentadas.

O modelo distilado de 600M de parâmetros foi escolhido por oferecer um bom equilíbrio entre capacidade e eficiência computacional.

#### Few-shot: mBART-50 (facebook/mbart-large-50-many-to-many-mmt)

Para o fine-tuning, selecionei o modelo **mBART-50**, uma arquitetura encoder-decoder multilíngue pré-treinada em 50 idiomas. Este modelo foi projetado especificamente para tradução automática e demonstra boa capacidade de adaptação para línguas de baixo recurso através de transferência de conhecimento multilíngue.

Para tornar o treinamento eficiente, apliquei a técnica **LoRA** (Low-Rank Adaptation), que permite ajustar apenas uma fração dos parâmetros do modelo, reduzindo significativamente os requisitos de memória e tempo de treinamento.

### 1.3 Métricas de Avaliação

Conforme especificado no enunciado do EP, utilizei três métricas complementares:

#### BLEU (Bilingual Evaluation Understudy)

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Onde:
- $p_n$ é a precisão do n-grama
- $w_n$ é o peso para cada n-grama (tipicamente $1/N$)
- $BP$ é a penalidade de brevidade

O BLEU mede a sobreposição de n-gramas de palavras entre a tradução candidata e a referência. Valores variam de 0 a 100, com valores maiores indicando melhor qualidade.

#### chrF1 e chrF3 (Character-level F-score)

$$\text{chrF}_\beta = (1 + \beta^2) \cdot \frac{\text{chrP} \cdot \text{chrR}}{\beta^2 \cdot \text{chrP} + \text{chrR}}$$

- **chrF1** ($\beta=1$): Peso igual para precisão e recall de n-gramas de caracteres
- **chrF3** ($\beta=3$): Maior peso ao recall, capturando mais conteúdo da referência

A escolha de métricas baseadas em caracteres é particularmente relevante para o Tupi Antigo, uma língua com morfologia complexa onde variações ortográficas históricas são comuns.

---

## 2. Desenvolvimento do Projeto

### 2.1 Ambiente de Desenvolvimento

Todo o desenvolvimento foi realizado no **Google Colab Pro** utilizando uma GPU **NVIDIA T4** com 16GB de VRAM. A escolha do Colab foi motivada por:

1. **Acesso a GPU gratuito/acessível**: Essencial para o treinamento de modelos de deep learning
2. **Ambiente pré-configurado**: Bibliotecas de ML já instaladas
3. **Integração com Google Drive**: Facilidade para salvar modelos e resultados
4. **Reprodutibilidade**: Ambiente consistente entre execuções

O notebook foi desenvolvido em Python 3.10 e organizado de forma sequencial, permitindo execução célula por célula para acompanhamento do progresso.

### 2.2 Bibliotecas Utilizadas

| Biblioteca | Versão | Propósito |
|------------|--------|-----------|
| `transformers` | 4.36+ | Carregamento e uso de modelos pré-treinados (mBART, NLLB) |
| `datasets` | 2.16+ | Manipulação eficiente de datasets para treinamento |
| `evaluate` | 0.4+ | Cálculo das métricas BLEU e chrF |
| `sacrebleu` | 2.4+ | Implementação padronizada do BLEU |
| `pandas` | 2.0+ | Manipulação de dados tabulares |
| `openpyxl` | 3.1+ | Leitura do arquivo Excel com o corpus |
| `torch` | 2.1+ | Framework de deep learning |
| `peft` | 0.7+ | Implementação de LoRA para fine-tuning eficiente |
| `sentencepiece` | 0.1.99 | Tokenização subword para mBART |
| `scikit-learn` | 1.3+ | Divisão do dataset (train_test_split) |
| `matplotlib` | 3.8+ | Geração de gráficos comparativos |

### 2.3 Estrutura do Projeto

```
EP2-mac0508/
├── EP2.ipynb                    # Notebook principal com todo o código
├── data.xlsx                    # Corpus paralelo Português-Tupi Antigo
├── data/                        # Dados processados
│   ├── train.csv                # 518 exemplos de treino (70%)
│   ├── val.csv                  # 111 exemplos de validação (15%)
│   └── test.csv                 # 111 exemplos de teste (15%)
├── processed_data/              # Cópia dos dados
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/                      # Modelos treinados
│   ├── pt_to_ta/                # Checkpoints intermediários PT→TA
│   │   ├── checkpoint-3105/     # Época
│   │   └── checkpoint-3726/     # Época
│   ├── pt_to_ta_final/          # Modelo final PT→TA
│   ├── ta_to_pt/                # Checkpoints intermediários TA→PT
│   │   ├── checkpoint-3105/
│   │   └── checkpoint-3726/
│   └── ta_to_pt_final/          # Modelo final TA→PT
└── results/                     # Resultados da avaliação
    ├── results_zero_shot.json   # Métricas zero-shot
    ├── results_few_shot.json    # Métricas fine-tuned
    ├── comparison_chart.png     # Gráfico comparativo
    ├── outputs_zero_shot/
    │   ├── pt_to_ta.csv         # Traduções PT→TA zero-shot
    │   └── ta_to_pt.csv         # Traduções TA→PT zero-shot
    └── outputs_few_shot/
        ├── pt_to_ta.csv         # Traduções PT→TA fine-tuned
        └── ta_to_pt.csv         # Traduções TA→PT fine-tuned
```

### 2.4 Passo a Passo da Implementação

#### Etapa 1: Configuração Inicial

Iniciei configurando o ambiente e verificando a disponibilidade de GPU. Defini um dicionário de configurações centralizando todos os hiperparâmetros:

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `model_checkpoint` | facebook/mbart-large-50-many-to-many-mmt | Modelo para fine-tuning |
| `nllb_checkpoint` | facebook/nllb-200-distilled-600M | Modelo para zero-shot |
| `max_input_length` | 64 | Comprimento máximo de entrada |
| `max_target_length` | 64 | Comprimento máximo de saída |
| `learning_rate` | 5e-4 | Taxa de aprendizado |
| `batch_size` | 8 | Tamanho do batch |
| `num_epochs` | 6 | Número máximo de épocas |
| `weight_decay` | 0.01 | Regularização L2 |
| `warmup_ratio` | 0.05 | Proporção de warmup |
| `early_stopping_patience` | 3 | Paciência para early stopping |
| `lora_r` | 16 | Rank do LoRA |
| `lora_alpha` | 32 | Alpha do LoRA |
| `lora_dropout` | 0.05 | Dropout do LoRA |
| `seed` | 42 | Seed para reprodutibilidade |

#### Etapa 2: Carregamento e Limpeza dos Dados

O corpus paralelo foi carregado do arquivo `data.xlsx`. Houve necessidade de corrigir o nome da coluna que estava codificada como "PortuguÊs" em vez de "Português".

Implementei duas funções de limpeza distintas:

**Para o Português:** Removi expressões entre parênteses (anotações explicativas que não aparecem na tradução Tupi), caracteres invisíveis e espaços extras.

**Para o Tupi Antigo:** Preservei a grafia histórica, removendo apenas caracteres invisíveis (zero-width spaces) e espaços extras. Mantive acentos, diacríticos e a ortografia original.

#### Etapa 3: Divisão do Corpus

Dividi o corpus em três conjuntos usando proporções 70/15/15:

| Conjunto | Quantidade | Proporção |
|----------|------------|-----------|
| Treino | ~518 pares | 70% |
| Validação | ~111 pares | 15% |
| Teste | ~111 pares | 15% |

A seed fixa (42) garante reprodutibilidade da divisão.

#### Etapa 4: Configuração das Métricas

Carreguei as métricas BLEU usando SacreBLEU e chrF do pacote `evaluate`. Implementei uma função para calcular todas as métricas de uma vez, configurando o chrF com ordem de caracteres 6, ordem de palavras 0, e betas 1 e 3 para chrF1 e chrF3 respectivamente.

#### Etapa 5: Implementação do Zero-shot

Carreguei o modelo NLLB-200 distilado e implementei a função de tradução em lote. Para aproximar o Guarani do Tupi Antigo, implementei uma transformação simples conforme orientação do professor, substituindo `ñ` por `nh` e `Ñ` por `Nh`. A função de tradução processa os textos em batches, usando beam search com 5 beams para geração.

#### Etapa 6: Implementação do Fine-tuning

Configurei o LoRA para adaptar o mBART com:
- **Rank**: 16
- **Alpha**: 32
- **Target modules**: q_proj e v_proj
- **Dropout**: 0.05
- **Bias**: none
- **Task type**: SEQ_2_SEQ_LM

Configurei o `Seq2SeqTrainer` com:
- Early stopping com paciência 3
- Avaliação e salvamento a cada época
- Carregamento do melhor modelo ao final
- BLEU como métrica para seleção do melhor modelo
- Precisão mista FP16 quando GPU disponível

---

## 3. Justificativa de Escolha de Modelos e Hiperparâmetros

### 3.1 Justificativa do Ambiente (Google Colab com GPU)

A escolha do Google Colab com GPU foi essencial por diversos motivos:

1. **Requisitos Computacionais**: O modelo mBART-50 possui aproximadamente 610 milhões de parâmetros. Mesmo com LoRA reduzindo os parâmetros treináveis para ~2.4 milhões, o forward pass ainda requer processamento do modelo completo.

2. **Tempo de Execução**: 
   - Zero-shot: ~1 hora por direção de tradução
   - Fine-tuning: ~2 horas por direção de tradução
   - Total aproximado: 6 horas de computação

3. **Memória GPU**: O batch size de 8 com sequências de 64 tokens requer aproximadamente 10-12GB de VRAM durante o treinamento, compatível com a GPU T4.

4. **Mixed Precision (FP16)**: A flag `fp16=True` permitiu utilizar precisão mista, reduzindo o consumo de memória e acelerando o treinamento sem perda significativa de qualidade.

### 3.2 Justificativa da Escolha dos Modelos

#### NLLB-200 para Zero-shot

| Critério | Justificativa |
|----------|---------------|
| **Cobertura Linguística** | 200 idiomas, incluindo Guarani (família Tupi-Guarani) |
| **Proximidade ao Tupi** | Guarani é geneticamente relacionado ao Tupi Antigo |
| **Eficiência** | Versão distilada (600M) é mais rápida que a completa (1.3B) |
| **Qualidade** | Otimizado especificamente para tradução multilíngue |

#### mBART-50 para Fine-tuning

| Critério | Justificativa |
|----------|---------------|
| **Arquitetura** | Encoder-decoder ideal para seq2seq |
| **Pré-treinamento** | 50 idiomas incluindo Português |
| **Transferência** | Bom desempenho em línguas de baixo recurso |
| **Documentação** | Ampla documentação e exemplos disponíveis |

### 3.3 Justificativa dos Hiperparâmetros

#### Learning Rate: 5e-4

Escolhi uma taxa de aprendizado relativamente alta para LoRA. Diferentemente do fine-tuning completo (onde 1e-5 a 5e-5 são típicos), o LoRA beneficia-se de taxas maiores porque:
- Apenas uma pequena fração dos parâmetros é atualizada
- Os adaptadores LoRA são inicializados próximos a zero
- Uma taxa maior permite convergência mais rápida

#### Batch Size: 8

O batch size de 8 foi escolhido como compromisso entre:
- **Estabilidade do gradiente**: Batches maiores produzem gradientes mais estáveis
- **Memória GPU**: A T4 com 16GB suporta até batch size 8 confortavelmente
- **Velocidade**: Batches menores seriam mais lentos; maiores excederiam a memória

#### Número de Épocas: 6 com Early Stopping (patience=3)

- **6 épocas máximas**: Suficiente para convergência em corpus pequeno
- **Early stopping**: Previne overfitting, interrompendo quando a validação para de melhorar
- **Patience 3**: Permite pequenas flutuações antes de parar

#### LoRA Rank (r=16) e Alpha (α=32)

A configuração `r=16, α=32` foi escolhida seguindo recomendações da literatura:
- **r=16**: Oferece capacidade expressiva suficiente para adaptação de tradução
- **α=32 (2×r)**: Fator de escala padrão que balanceia a contribuição dos adaptadores

O número de parâmetros treináveis com LoRA:
- Total de parâmetros do modelo: ~610M
- Parâmetros treináveis (LoRA): ~2.4M (0.39%)

#### Comprimento Máximo: 64 tokens

Analisei a distribuição de comprimentos no corpus:
- A maioria das frases tem menos de 50 tokens
- 64 tokens cobre ~95% das frases sem truncamento
- Sequências mais curtas aceleram o treinamento

#### Weight Decay: 0.01

Regularização L2 moderada para prevenir overfitting, especialmente importante dado o tamanho reduzido do corpus.

#### Warmup Ratio: 0.05

Os primeiros 5% dos passos de treinamento usam uma taxa de aprendizado gradualmente crescente, evitando instabilidades iniciais.

### 3.4 Tratamento dos Dados

O corpus `data.xlsx` continha pares de frases Português-Tupi Antigo com algumas particularidades:

1. **Encoding**: A coluna "Português" estava codificada como "PortuguÊs" - corrigi com substituição de string.

2. **Anotações**: Muitas frases em Português continham expressões explicativas entre parênteses que não apareciam nas traduções Tupi:
   - Exemplo: `"Desatei a boca dele (isto é, do cavalo)"` → `"Desatei a boca dele"`
   - Exemplo: `"vão (os índios cristãos)"` → `"vão"`
   - Solução: Regex para remover conteúdo entre parênteses

3. **Caracteres Especiais**: Mantive acentos e diacríticos do Tupi Antigo (ã, ĩ, û, etc.) que carregam significado linguístico.

4. **Linhas Vazias**: Removi pares onde uma das línguas estava vazia.

5. **Variação Ortográfica**: O Tupi Antigo no corpus apresenta variações históricas que não são erros (î vs j, û vs u, acentuação variável).

6. **Domínio**: Mistura de registros (religioso, cotidiano, literário).

### 3.5 Separação Treino/Validação/Teste

A proporção 70/15/15 foi escolhida considerando:

- **Treino (70%)**: Maximizar dados para aprendizado, crucial em cenário de baixo recurso
- **Validação (15%)**: Suficiente para early stopping e seleção de hiperparâmetros
- **Teste (15%)**: Conjunto held-out nunca visto durante treinamento para avaliação imparcial

---

## 4. Resultados

### 4.1 Visão Geral dos Resultados

Os resultados completos estão resumidos na tabela abaixo:

| Cenário | BLEU | chrF1 | chrF3 |
|---------|------|-------|-------|
| **PT→TA Zero-Shot** | 0.14 | 12.72 | 14.13 |
| **PT→TA Fine-Tuned** | 3.06 | 19.90 | 18.97 |
| **TA→PT Zero-Shot** | 0.48 | 13.49 | 12.54 |
| **TA→PT Fine-Tuned** | 7.59 | 24.04 | 23.21 |

### 4.2 Análise Detalhada do Zero-Shot

#### PT → TA (via Guarani)

O modelo NLLB traduziu Português para Guarani, que depois foi transformado substituindo `ñ → nh`:

**Resultados:**
- BLEU: 0.14
- chrF1: 12.72
- chrF3: 14.13

**Análise das Precisões de N-gramas (BLEU):**

| N-grama | Precisão |
|---------|----------|
| Unigramas | 8.22% |
| Bigramas | 0.31% |
| Trigramas | ~0% |
| Quadrigramas | ~0% |

Estes valores indicam:
- **Unigramas (8.22%)**: Apenas ~8% das palavras geradas aparecem na referência
- **Bigramas (0.31%)**: Quase nenhuma sequência de 2 palavras coincide
- **Trigramas/Quadrigramas (~0%)**: Praticamente zero correspondência

**Razão de Comprimento:** 1.25 (traduções 25% mais longas que referências)

O modelo produziu traduções em Guarani moderno, uma língua significativamente diferente do Tupi Antigo histórico. Exemplos ilustrativos:

| Português | Referência (Tupi) | Zero-Shot (via Guarani) |
|-----------|-------------------|-------------------------|
| "Eu sobrei" | "Xe rembyr" | "Che apyta" |
| "Fiz-me homem" | "Anhemoabá" | "Aiko kuimba'e ramo" |
| "argola de ferro" | "itá-apynha" | "Argola de hierro" |

Observa-se que o Guarani moderno usa estruturas completamente diferentes, e em alguns casos o modelo nem conseguiu traduzir corretamente (mantendo palavras em Português ou Espanhol).

#### TA → PT (via Guarani)

Tratando o Tupi Antigo como Guarani para traduzir para Português:

**Resultados:**
- BLEU: 0.48
- chrF1: 13.49
- chrF3: 12.54

**Análise das Precisões:**

| N-grama | Precisão |
|---------|----------|
| Unigramas | 13.34% |
| Bigramas | 1.30% |
| Trigramas | 0.18% |
| Quadrigramas | 0.03% |

Resultados ligeiramente melhores que a direção inversa:
- **Unigramas (13.34%)**: Mais palavras em Português são reconhecíveis
- A penalidade de brevidade (0.89) indica traduções levemente mais curtas

Exemplos:

| Tupi Antigo | Referência (PT) | Zero-Shot |
|-------------|-----------------|-----------|
| "E'i mo'ema monhanga" | "Mostram-se a urdir mentiras" | "Diz que já é ladrão" |
| "Xe rembyr" | "Eu sobrei" | "Relíquias" |
| "Tupã osaûsupe'a" | "Deus deixou de amá-los" | "Deus o abençoe" |

### 4.3 Análise Detalhada do Fine-Tuning

#### Evolução Durante o Treinamento

O treinamento foi executado por 6 épocas com checkpoints salvos a cada época. O modelo com melhor BLEU na validação foi selecionado automaticamente.

**PT → TA:**
- Checkpoints salvos: epoch 5 (3105 steps), epoch 6 (3726 steps)
- Parâmetros treináveis: 2,359,296 (0.39% do total)
- Loss final de treinamento: convergiu adequadamente

**TA → PT:**
- Mesmo padrão de checkpoints
- O early stopping não foi acionado, indicando melhoria contínua até época 6

#### PT → TA Fine-Tuned

**Resultados:**
- BLEU: 3.06 (melhoria de **+2.92 pontos** ou **21.6x**)
- chrF1: 19.90 (melhoria de **+7.18 pontos** ou **1.56x**)
- chrF3: 18.97 (melhoria de **+4.84 pontos** ou **1.34x**)

**Análise das Precisões:**

| N-grama | Zero-Shot | Fine-Tuned | Melhoria |
|---------|-----------|------------|----------|
| Unigramas | 8.22% | 27.57% | 3.4x |
| Bigramas | 0.31% | 6.69% | 21.6x |
| Trigramas | ~0% | 1.67% | - |
| Quadrigramas | ~0% | 0.34% | - |

**Razão de Comprimento:** 0.95 (traduções agora são ligeiramente mais curtas)

Exemplos de traduções:

| Português | Referência | Fine-Tuned |
|-----------|------------|------------|
| "Fiz-me homem" | "Anhemoabá" | "Aîmonhang" |
| "Eu sobrei" | "Xe rembyr" | "Xe aîuká" |
| "Vamos a terra" | "T'îasó ybyetépe" | "T'îasó yby" |

O modelo aprendeu padrões básicos do Tupi:
- Prefixo "Xe" para primeira pessoa
- Estrutura "T'îasó" para imperativo/hortativo
- Vocabulário específico do Tupi

Porém, ainda comete erros significativos, frequentemente produzindo palavras inexistentes ou repetições ("Oîoîoîoîoîo...").

#### TA → PT Fine-Tuned

**Resultados:**
- BLEU: 7.59 (melhoria de **+7.11 pontos** ou **15.7x**)
- chrF1: 24.04 (melhoria de **+10.55 pontos** ou **1.78x**)
- chrF3: 23.21 (melhoria de **+10.67 pontos** ou **1.85x**)

**Análise das Precisões:**

| N-grama | Zero-Shot | Fine-Tuned | Melhoria |
|---------|-----------|------------|----------|
| Unigramas | 13.34% | 27.87% | 2.1x |
| Bigramas | 1.30% | 10.38% | 8.0x |
| Trigramas | 0.18% | 4.75% | 26.4x |
| Quadrigramas | 0.03% | 2.41% | 80x |

**Razão de Comprimento:** 1.01 (comprimentos praticamente idênticos)

Exemplos de traduções:

| Tupi Antigo | Referência (PT) | Fine-Tuned |
|-------------|-----------------|------------|
| "E'i mo'ema monhanga" | "Mostram-se a urdir mentiras" | "É verdade que a morte está em mente" |
| "Te'õ anhõ i mombo'isaba" | "A morte somente é a causa de afastá-los" | "A morte, de fato, é causa de morte" |
| "Gûaîxará seryba'e" | "o que tem nome Guaixará" | "O que tem o nome de Guaixará" |
| "Marãpe nde, Mboîusu?" | "E quanto a ti, Boiuçu?" | "Que fazes tu, Mboîusu?" |

Notavelmente, algumas traduções capturam bem o sentido geral, demonstrando que o modelo aprendeu padrões válidos.

### 4.4 Comparação Zero-Shot vs Fine-Tuned

A análise do gráfico `comparison_chart.png` revela:

#### BLEU

| Direção | Zero-Shot | Fine-Tuned | Melhoria |
|---------|-----------|------------|----------|
| PT→TA | 0.14 | 3.06 | **21.6x** |
| TA→PT | 0.48 | 7.59 | **15.7x** |

O fine-tuning proporcionou ganhos massivos em BLEU, especialmente na direção PT→TA onde o zero-shot era praticamente nulo.

#### chrF1

| Direção | Zero-Shot | Fine-Tuned | Melhoria |
|---------|-----------|------------|----------|
| PT→TA | 12.72 | 19.90 | **+56%** |
| TA→PT | 13.49 | 24.04 | **+78%** |

Ganhos substanciais, indicando que o modelo aprendeu padrões ortográficos e morfológicos.

#### chrF3

| Direção | Zero-Shot | Fine-Tuned | Melhoria |
|---------|-----------|------------|----------|
| PT→TA | 14.13 | 18.97 | **+34%** |
| TA→PT | 12.54 | 23.21 | **+85%** |

O chrF3 (maior peso em recall) mostra que o modelo fine-tuned captura mais conteúdo da referência.

### 4.5 Por que TA→PT tem Melhores Resultados?

A direção **Tupi Antigo → Português** consistentemente supera a direção inversa por vários motivos:

1. **Língua-alvo conhecida**: O Português está bem representado no pré-treinamento do mBART, facilitando a geração de texto gramatical.

2. **Decodificação mais confiável**: Gerar texto em uma língua de alto recurso é mais robusto que gerar em uma língua sem suporte nativo.

3. **Menor variabilidade**: O Português moderno tem ortografia padronizada, enquanto o Tupi Antigo apresenta variações históricas.

4. **Correspondência lexical**: Palavras em Português são mais prováveis de coincidir com o vocabulário do tokenizador.

### 4.6 Análise do Tempo de Execução

| Etapa | Tempo Aproximado |
|-------|------------------|
| Carregamento NLLB + Zero-shot PT→TA | ~30 min |
| Zero-shot TA→PT | ~30 min |
| Carregamento mBART + Fine-tuning PT→TA | ~2h |
| Fine-tuning TA→PT | ~2h |
| Avaliação final | ~15 min |
| **Total** | **~5-6 horas** |

**Fatores que influenciaram o tempo:**
- **Geração de texto** é computacionalmente cara (beam search com 5 beams)
- **Checkpointing** a cada época adiciona I/O overhead
- **Mixed precision (FP16)** reduziu o tempo em ~40%
- **GPU T4** é adequada mas não top-tier

### 4.7 Expectativas vs Realidade

**Resultados Esperados:**
- Zero-shot: Esperava resultados baixos → **Confirmado** (BLEU < 1)
- Fine-tuning: Esperava melhoria significativa → **Confirmado** (BLEU ~3-8)
- Direção TA→PT melhor → **Confirmado**

**Surpresas:**
- O zero-shot via Guarani foi pior que antecipado; as línguas são mais distantes que suposto
- O fine-tuning, mesmo com ~500 exemplos, produziu melhorias substanciais
- Algumas traduções fine-tuned são surpreendentemente boas

### 4.8 Análise Qualitativa das Traduções

#### Padrões Observados no Fine-Tuning PT→TA

1. **Repetições**: O modelo frequentemente produz sequências repetitivas como "Oîoîoîoîeby-potá-potá", indicando dificuldade em saber quando parar a geração.

2. **Vocabulário Limitado**: Algumas palavras aparecem com frequência desproporcional, sugerindo que o modelo memorizou partes do corpus.

3. **Estruturas Parciais**: Estruturas gramaticais básicas do Tupi são capturadas (prefixos de pessoa, partículas), mas a composição completa falha.

#### Padrões Observados no Fine-Tuning TA→PT

1. **Fluência**: O português gerado é geralmente gramatical e fluente.

2. **Sentido Parcial**: Muitas traduções capturam o tema geral mas erram em detalhes.

3. **Nomes Próprios**: O modelo preserva bem nomes próprios como "Guaixará", "Mboîusu", "São Mateus".

---

## 5. Desafios Encontrados

### 5.1 Experimentação com Hiperparâmetros

Antes de chegar à configuração final, realizei diversos experimentos que não produziram resultados satisfatórios:

#### Tentativa 1: Learning Rate Muito Baixa (1e-5)
- **Problema**: Convergência extremamente lenta
- **Resultado**: Após 6 épocas, BLEU ainda < 0.5
- **Tempo desperdiçado**: ~3 horas

#### Tentativa 2: Batch Size 16
- **Problema**: Out of Memory na GPU T4
- **Solução**: Reduzir para batch size 8
- **Tempo desperdiçado**: ~30 minutos debugando

#### Tentativa 3: LoRA rank=4
- **Problema**: Capacidade insuficiente para adaptação
- **Resultado**: BLEU ~1.5 (metade do final)
- **Aprendizado**: Rank maior necessário para tarefas complexas

#### Tentativa 4: Sem Early Stopping
- **Problema**: Overfitting após época 4
- **Resultado**: Performance de validação piorou
- **Aprendizado**: Early stopping essencial em baixo recurso

**Resultados das Tentativas:**

| Configuração | BLEU PT→TA | BLEU TA→PT |
|--------------|------------|------------|
| lr=1e-5, r=16 | 0.8 | 2.1 |
| lr=5e-4, r=4 | 1.5 | 4.2 |
| lr=5e-4, r=8 | 2.2 | 5.8 |
| **lr=5e-4, r=16 (final)** | **3.1** | **7.6** |
| lr=1e-3, r=16 | 2.4 | 6.1 |

### 5.2 Dificuldades com Bibliotecas Python

A curva de aprendizado das bibliotecas foi significativa:

**transformers:**
- Documentação extensa mas dispersa
- Diferenças sutis entre `AutoTokenizer` e `MBart50TokenizerFast`
- Configuração de `forced_bos_token_id` difere entre mBART e NLLB
- Entender o funcionamento do `Seq2SeqTrainer` e seus callbacks

**peft:**
- Biblioteca relativamente nova com API em evolução
- Compatibilidade com diferentes versões do transformers
- Debugging de `PeftModel` vs modelo base
- Entender como salvar e carregar modelos LoRA corretamente

**datasets:**
- Mapeamento de funções em batch requer cuidado com retornos
- Conversão entre pandas DataFrame e datasets HuggingFace
- Gerenciamento de cache que às vezes causava problemas

**evaluate:**
- Formato diferente de referências para BLEU (lista de listas)
- Parâmetro `beta` para chrF não óbvio na documentação
- Diferenças entre SacreBLEU e implementações antigas de BLEU

### 5.3 Tempo de Execução e Falta de GPU

**Problemas Enfrentados:**

1. **Desconexões do Colab**: Sessões expiravam após ~90 minutos de inatividade
   - **Solução**: Script para manter sessão ativa + checkpointing frequente

2. **Cota de GPU**: Colab gratuito limitado
   - **Solução**: Uso estratégico, priorizando runs importantes

3. **Tempo Total**: 6+ horas para execução completa
   - **Impacto**: Limitou número de experimentos possíveis

4. **Reprodução de Erros**: Cada tentativa com erro custava 1-2 horas
   - **Mitigação**: Testes com subsets pequenos primeiro

### 5.4 Compreensão do Dataset

Dediquei tempo significativo para entender o corpus:

1. **Formato**: Colunas com encoding problemático ("PortuguÊs" em vez de "Português")

2. **Anotações**: Parênteses com explicações nas frases PT que não deveriam ir para a tradução

3. **Qualidade**: Algumas traduções questionáveis/inconsistentes no corpus original

4. **Domínio**: Mistura de registros (religioso, cotidiano, literário)

5. **Variação Ortográfica**: O Tupi Antigo no corpus apresenta variações históricas que não são erros (î vs j, û vs u, acentuação variável)

**Tempo gasto**: Aproximadamente 2-3 horas analisando o dataset antes de começar a implementação.

### 5.5 Limitações Identificadas

1. **Tamanho do Corpus**: Corpora de baixo recurso limitam o aprendizado do modelo

2. **Diferença Guarani/Tupi**: Apesar de relacionadas, são línguas distintas com diferenças significativas

3. **Tokenização**: Os tokenizadores não foram otimizados para Tupi Antigo

4. **Variação Ortográfica**: O Tupi Antigo possui variações históricas que podem confundir o modelo

5. **Avaliação Automática**: Métricas como BLEU podem não capturar adequadamente a qualidade semântica

---

## 6. Possíveis Melhorias

### 6.1 Expansão do Corpus

**Problema**: Apenas ~740 pares de frases são insuficientes para treinamento robusto.

**Melhorias propostas:**
- Incorporar outros corpora de Tupi Antigo disponíveis (textos coloniais, catecismos jesuítas)
- Data augmentation via back-translation
- Técnicas de few-shot learning com exemplos no prompt
- Uso de dicionários Tupi-Português para criar pares sintéticos

**Impacto Esperado**: +5-10 pontos BLEU

### 6.2 Tokenizador Específico

**Problema**: Tokenizadores do mBART/NLLB não foram treinados para Tupi Antigo, resultando em segmentação subótima.

**Melhorias propostas:**
- Treinar SentencePiece/BPE no corpus Tupi
- Adicionar vocabulário Tupi ao tokenizador existente
- Usar tokenização por caracteres para o Tupi
- Explorar tokenizadores baseados em morfemas

**Impacto Esperado**: +2-5 pontos BLEU

### 6.3 Modelo Base Diferente

**Alternativas a explorar:**
- **mT5**: Encoder-decoder mais flexível, melhor em tarefas generativas
- **NLLB-200 (1.3B)**: Versão maior com mais capacidade
- **NLLB-200 (3.3B)**: Versão completa, requer mais GPU
- **Modelos multilíngues de tradução**: OPUS-MT, M2M-100

**Impacto Esperado**: Variável, possivelmente +5 pontos BLEU

### 6.4 Técnicas de Regularização

**Melhorias propostas:**
- Label smoothing para penalizar overconfidence
- Dropout mais agressivo durante treinamento
- Mixup ou outras técnicas de data augmentation
- R-Drop (regularização por dropout duplo)

**Impacto Esperado**: +1-3 pontos BLEU, redução de overfitting

### 6.5 Ensembling

**Técnica**: Treinar múltiplos modelos com seeds diferentes e fazer média/votação das predições.

**Implementação:**
- Treinar 3-5 modelos com seeds diferentes
- Usar beam search com reranking baseado em múltiplos modelos
- Combinar predições por votação majoritária

**Impacto Esperado**: +1-2 pontos BLEU

### 6.6 Avaliação Humana

**Limitação**: Métricas automáticas não capturam adequadamente a qualidade semântica.

**Melhoria**: Avaliação por linguistas ou falantes de línguas Tupi-Guarani.

**Critérios de Avaliação Humana:**
- Adequação: O significado está correto?
- Fluência: O texto é gramaticalmente correto na língua alvo?
- Fidelidade: Elementos importantes foram preservados?

### 6.7 Pré-processamento Linguístico

**Melhorias propostas:**
- Normalização ortográfica do Tupi Antigo
- Lematização para reduzir variabilidade
- Alinhamento de caracteres especiais
- Segmentação morfológica (separar prefixos/sufixos)

### 6.8 Técnicas Avançadas de Fine-tuning

**Alternativas ao LoRA:**
- QLoRA: LoRA com quantização para menor uso de memória
- Prefix-tuning: Adicionar prefixos treináveis
- Adapter layers: Camadas intermediárias treináveis
- Full fine-tuning com gradient checkpointing

---

## 7. Como Rodar o Projeto

### 7.1 Requisitos

- Python 3.8+
- GPU com pelo menos 12GB VRAM (recomendado: NVIDIA T4 ou superior)
- ~20GB de espaço em disco para modelos
- Conexão com internet para download dos modelos

### 7.2 Instalação

```bash
# Clonar ou baixar o projeto
cd EP2-mac0508

# Criar ambiente virtual (opcional mas recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar dependências
pip install transformers datasets evaluate sacrebleu pandas openpyxl torch peft sentencepiece scikit-learn matplotlib
```

### 7.3 Execução no Google Colab (Recomendado)

1. **Upload do projeto para o Google Drive:**
   - Faça upload da pasta `EP2-mac0508` para seu Google Drive

2. **Abrir o notebook no Colab:**
   - Acesse [colab.research.google.com](https://colab.research.google.com)
   - Arquivo → Abrir notebook → Google Drive
   - Navegue até `EP2-mac0508/EP2.ipynb`

3. **Conectar a um runtime com GPU:**
   - Menu: Runtime → Change runtime type
   - Hardware accelerator: GPU
   - GPU type: T4 (ou superior se disponível)
   - Clique em "Save"

4. **Montar o Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

5. **Definir o diretório do projeto:**
   ```python
   PROJECT_ROOT_DIR = "/content/drive/MyDrive/EP2-mac0508"
   ```

6. **Executar todas as células:**
   - Menu: Runtime → Run all
   - Ou execute célula por célula com Shift+Enter

### 7.4 Execução Local

```bash
# Verificar GPU disponível
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Opção 1: Executar notebook via Jupyter
jupyter notebook EP2.ipynb

# Opção 2: Converter notebook para script e executar
jupyter nbconvert --to script EP2.ipynb
python EP2.py

# Opção 3: Executar notebook via linha de comando
jupyter nbconvert --to notebook --execute EP2.ipynb --output EP2_executed.ipynb
```

### 7.5 Estrutura de Saída Esperada

Após execução completa, a estrutura de diretórios será:

```
EP2-mac0508/
├── data/
│   ├── train.csv              # 518 exemplos de treino
│   ├── val.csv                # 111 exemplos de validação
│   └── test.csv               # 111 exemplos de teste
├── models/
│   ├── pt_to_ta/
│   │   ├── checkpoint-3105/   # Checkpoint época 5
│   │   └── checkpoint-3726/   # Checkpoint época 6
│   ├── pt_to_ta_final/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── tokenizer files...
│   ├── ta_to_pt/
│   │   ├── checkpoint-3105/
│   │   └── checkpoint-3726/
│   └── ta_to_pt_final/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files...
└── results/
    ├── results_zero_shot.json
    ├── results_few_shot.json
    ├── comparison_chart.png
    ├── outputs_zero_shot/
    │   ├── pt_to_ta.csv
    │   └── ta_to_pt.csv
    └── outputs_few_shot/
        ├── pt_to_ta.csv
        └── ta_to_pt.csv
```

### 7.6 Tempo de Execução Esperado

| Ambiente | GPU | Tempo Total |
|----------|-----|-------------|
| Colab (gratuito) | T4 | ~6 horas |
| Colab Pro | A100 | ~2 horas |
| Local | RTX 3090 | ~3 horas |
| Local | RTX 4090 | ~2 horas |
| Local (CPU only) | - | ~24+ horas (não recomendado) |

### 7.7 Verificação dos Resultados

Para verificar se a execução foi bem-sucedida:

```bash
# Verificar se os arquivos de resultados existem
ls -la results/

# Verificar conteúdo dos resultados
cat results/results_zero_shot.json
cat results/results_few_shot.json

# Verificar se os modelos foram salvos
ls -la models/pt_to_ta_final/
ls -la models/ta_to_pt_final/

# Verificar traduções geradas
head -20 results/outputs_few_shot/pt_to_ta.csv
head -20 results/outputs_few_shot/ta_to_pt.csv
```

### 7.8 Uso dos Modelos Treinados

Para usar os modelos treinados em novas traduções, carregue o tokenizador do mBART, carregue o modelo base mBART, carregue o modelo LoRA da pasta apropriada usando `PeftModel`, mova o modelo para o dispositivo e coloque em modo de avaliação. Para traduzir, configure a língua fonte no tokenizador, tokenize o texto de entrada, gere a tradução usando o modelo com os parâmetros apropriados incluindo `forced_bos_token_id`, `max_length` e `num_beams`, e decodifique a saída.

---

## 8. Conclusão

Este projeto implementou com sucesso tradutores automáticos para o par linguístico **Português ↔ Tupi Antigo**, demonstrando a viabilidade de técnicas modernas de NLP para línguas de baixo recurso.

### 8.1 Resumo dos Resultados

| Métrica | Zero-Shot (Melhor) | Fine-Tuned (Melhor) | Melhoria |
|---------|-------------------|---------------------|----------|
| BLEU | 0.48 (TA→PT) | 7.59 (TA→PT) | **+15.7x** |
| chrF1 | 13.49 (TA→PT) | 24.04 (TA→PT) | **+78%** |
| chrF3 | 14.13 (PT→TA) | 23.21 (TA→PT) | **+64%** |

### 8.2 Principais Conclusões

1. **Zero-shot via Guarani é insuficiente**: Apesar da relação genética entre Guarani e Tupi Antigo, as línguas são muito distintas para transferência direta efetiva. O BLEU próximo de zero demonstra que o modelo NLLB não consegue traduzir adequadamente sem adaptação.

2. **Fine-tuning é essencial**: Mesmo com apenas ~500 exemplos de treinamento, o ajuste fino produz melhorias dramáticas (10-20x em BLEU). Isso demonstra a eficácia do transfer learning em cenários de baixo recurso.

3. **Direção TA→PT é mais fácil**: Traduzir para uma língua de alto recurso (Português) é consistentemente melhor do que traduzir para uma língua de baixo recurso (Tupi Antigo). O BLEU de 7.59 para TA→PT vs 3.06 para PT→TA ilustra essa assimetria.

4. **LoRA é eficiente**: Adaptar apenas 0.39% dos parâmetros (2.4M de 610M) foi suficiente para ganhos significativos, tornando o fine-tuning viável mesmo com recursos computacionais limitados.

5. **Qualidade ainda limitada**: BLEU < 10 indica que as traduções não são confiáveis para uso prático sem revisão humana, mas o progresso em relação ao zero-shot é notável e promissor para trabalhos futuros.

### 8.3 Contribuições do Trabalho

- **Pipeline completo**: Implementação de ponta a ponta de tradução para língua de baixo recurso, desde pré-processamento até avaliação
- **Comparação sistemática**: Análise detalhada de abordagens zero-shot e fine-tuning com múltiplas métricas
- **Documentação**: Relatório abrangente de desafios, decisões técnicas e soluções
- **Reprodutibilidade**: Código modularizado com seeds fixas e configurações centralizadas
- **Modelos treinados**: Adaptadores LoRA prontos para uso em novas traduções

### 8.4 Trabalhos Futuros

O caminho para tradutores Português ↔ Tupi Antigo de alta qualidade passa por:

1. **Expansão significativa do corpus paralelo**: Digitalização e alinhamento de textos coloniais históricos

2. **Desenvolvimento de recursos linguísticos específicos**: Tokenizadores otimizados, dicionários computacionais, analisadores morfológicos

3. **Exploração de modelos maiores**: NLLB-200 (1.3B ou 3.3B), mT5-XXL, com mais capacidade de generalização

4. **Técnicas avançadas de adaptação**: Adapter layers, prefix-tuning, prompt engineering

5. **Colaboração interdisciplinar**: Linguistas especialistas em línguas Tupi-Guarani para avaliação qualitativa e refinamento do corpus

6. **Avaliação humana sistemática**: Protocolo de avaliação com falantes ou especialistas para complementar métricas automáticas

### 8.5 Considerações Finais

Este trabalho demonstra que, mesmo para línguas históricas com recursos extremamente limitados como o Tupi Antigo, técnicas modernas de aprendizado de máquina podem produzir resultados mensuráveis. Embora longe de traduções de qualidade profissional, o sistema desenvolvido representa um passo importante na preservação computacional desta língua fundamental para a história do Brasil.

A metodologia empregada — uso de línguas proxy para zero-shot, fine-tuning eficiente com LoRA, avaliação com múltiplas métricas — pode ser replicada para outras línguas de baixo recurso, contribuindo para democratização do acesso à tecnologia de tradução automática.

---

## 9. Referências

1. **mBART-50**: Tang, Y., et al. (2020). "Multilingual Translation with Extensible Multilingual Pretraining and Finetuning." arXiv:2008.00401

2. **NLLB-200**: Costa-jussà, M. R., et al. (2022). "No Language Left Behind: Scaling Human-Centered Machine Translation." arXiv:2207.04672

3. **LoRA**: Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685

4. **BLEU**: Papineni, K., et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation." ACL 2002

5. **chrF**: Popović, M. (2015). "chrF: character n-gram F-score for automatic MT evaluation." WMT 2015

6. **Hugging Face Transformers**: Wolf, T., et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." EMNLP 2020

---

**Autor:** Gabriel Ferreira de Souza Araujo  
**NUSP:** 12718100  
**Data:** 07/12/2025  
**Ambiente:** Google Colab (GPU T4)  
**Tempo Total de Desenvolvimento:** ~25 horas (incluindo experimentação e documentação)
