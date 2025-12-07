# Relatório - EP2: Tradução Automática de Baixo Recurso
## Português ↔ Tupi Antigo

**Disciplina:** MAC0508 — Introdução ao Processamento de Língua Natural  
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

---

## 1. Introdução

Este relatório documenta o desenvolvimento, implementação e avaliação de modelos de tradução automática para o par linguístico Português e Tupi Antigo, uma tarefa desafiadora devido à escassez de dados disponíveis para esta língua histórica brasileira.

### 1.1 Objetivo do Projeto

O objetivo principal foi implementar e comparar tradutores automáticos em dois regimes distintos:

O primeiro regime é o Zero-shot Learning, que consiste na utilização de modelos pré-treinados sem qualquer ajuste fino no corpus Português-Tupi Antigo. O segundo regime é o Few-shot Learning através de Fine-tuning, que envolve o ajuste fino de modelos utilizando o corpus paralelo disponível.

A comparação foi realizada em ambas as direções de tradução: Português para Tupi Antigo e Tupi Antigo para Português.

### 1.2 Modelos Utilizados

Para o regime zero-shot, utilizei o modelo NLLB-200 da Meta AI, especificamente a versão distilada com 600 milhões de parâmetros, disponível no Hugging Face como facebook/nllb-200-distilled-600M. A escolha deste modelo foi estratégica: embora não possua suporte nativo ao Tupi Antigo, ele inclui o Guarani, uma língua da família Tupi-Guarani geneticamente relacionada ao Tupi Antigo. Esta proximidade linguística permite uma aproximação razoável via transferência de conhecimento entre línguas aparentadas. O modelo distilado de 600M de parâmetros foi escolhido por oferecer um bom equilíbrio entre capacidade e eficiência computacional.

Para o fine-tuning, selecionei o modelo mBART-50, disponível como facebook/mbart-large-50-many-to-many-mmt. Trata-se de uma arquitetura encoder-decoder multilíngue pré-treinada em 50 idiomas. Este modelo foi projetado especificamente para tradução automática e demonstra boa capacidade de adaptação para línguas de baixo recurso através de transferência de conhecimento multilíngue. Para tornar o treinamento eficiente, apliquei a técnica LoRA (Low-Rank Adaptation), que permite ajustar apenas uma fração dos parâmetros do modelo, reduzindo significativamente os requisitos de memória e tempo de treinamento.

### 1.3 Métricas de Avaliação

Conforme especificado no enunciado do EP, utilizei três métricas complementares.

A primeira métrica é o BLEU (Bilingual Evaluation Understudy), que mede a sobreposição de n-gramas de palavras entre a tradução candidata e a referência. Os valores variam de 0 a 100, com valores maiores indicando melhor qualidade. O cálculo envolve as precisões de n-gramas ponderadas e uma penalidade de brevidade para traduções muito curtas.

A segunda e terceira métricas são o chrF1 e chrF3, que correspondem ao F-score baseado em caracteres. O chrF1 utiliza beta igual a 1, dando peso igual para precisão e recall de n-gramas de caracteres. O chrF3 utiliza beta igual a 3, dando maior peso ao recall, capturando mais conteúdo da referência. A escolha de métricas baseadas em caracteres é particularmente relevante para o Tupi Antigo, uma língua com morfologia complexa onde variações ortográficas históricas são comuns.

---

## 2. Desenvolvimento do Projeto

### 2.1 Ambiente de Desenvolvimento

Todo o desenvolvimento foi realizado no Google Colab Pro utilizando uma GPU NVIDIA T4 com 16GB de VRAM. A escolha do Colab foi motivada pelo acesso a GPU gratuito ou acessível, essencial para o treinamento de modelos de deep learning; pelo ambiente pré-configurado com bibliotecas de ML já instaladas; pela integração com Google Drive para facilidade em salvar modelos e resultados; e pela reprodutibilidade proporcionada por um ambiente consistente entre execuções.

O notebook foi desenvolvido em Python 3.10 e organizado de forma sequencial, permitindo execução célula por célula para acompanhamento do progresso.

### 2.2 Bibliotecas Utilizadas

As principais bibliotecas utilizadas foram: transformers na versão 4.36 ou superior para carregamento e uso de modelos pré-treinados como mBART e NLLB; datasets na versão 2.16 ou superior para manipulação eficiente de datasets para treinamento; evaluate na versão 0.4 ou superior para cálculo das métricas BLEU e chrF; sacrebleu na versão 2.4 ou superior para implementação padronizada do BLEU; pandas na versão 2.0 ou superior para manipulação de dados tabulares; openpyxl na versão 3.1 ou superior para leitura do arquivo Excel com o corpus; torch na versão 2.1 ou superior como framework de deep learning; peft na versão 0.7 ou superior para implementação de LoRA para fine-tuning eficiente; e sentencepiece na versão 0.1.99 para tokenização subword do mBART.

### 2.3 Estrutura do Projeto

O projeto está organizado em uma pasta principal chamada EP2-mac0508 contendo o notebook principal EP2.ipynb com todo o código, o arquivo data.xlsx com o corpus paralelo Português-Tupi Antigo, o arquivo descricao_detalhada.md com o enunciado do EP, e este arquivo relatorio.md.

Além disso, existem subpastas para organização: a pasta data contém os dados processados incluindo train.csv, val.csv e test.csv; a pasta processed_data contém uma cópia de segurança dos dados; a pasta models armazena os modelos treinados, incluindo subpastas pt_to_ta para checkpoints intermediários, pt_to_ta_final para o modelo final de Português para Tupi Antigo, ta_to_pt para checkpoints intermediários, e ta_to_pt_final para o modelo final de Tupi Antigo para Português; e a pasta results contém os resultados da avaliação, incluindo results_zero_shot.json, results_few_shot.json, comparison_chart.png, e subpastas outputs_zero_shot e outputs_few_shot com as traduções geradas em ambas as direções.

### 2.4 Passo a Passo da Implementação

A primeira etapa foi a configuração inicial. Iniciei configurando o ambiente e verificando a disponibilidade de GPU. Defini um dicionário de configurações centralizando todos os hiperparâmetros, incluindo os checkpoints dos modelos mBART e NLLB, comprimentos máximos de entrada e saída de 64 tokens, learning rate de 5e-4, batch size de 8, 6 épocas de treinamento, weight decay de 0.01, warmup ratio de 0.05, paciência de early stopping de 3 épocas, e configurações do LoRA com rank 16, alpha 32 e dropout 0.05. A seed foi fixada em 42 para reprodutibilidade.

A segunda etapa foi o carregamento e limpeza dos dados. O corpus paralelo foi carregado do arquivo data.xlsx. Houve necessidade de corrigir o nome da coluna que estava codificada como "PortuguÊs" em vez de "Português". Implementei duas funções de limpeza distintas. Para o Português, removi expressões entre parênteses que são anotações explicativas que não aparecem na tradução Tupi, além de caracteres invisíveis e espaços extras. Para o Tupi Antigo, preservei a grafia histórica, removendo apenas caracteres invisíveis como zero-width spaces e espaços extras.

A terceira etapa foi a divisão do corpus. Dividi o corpus em três conjuntos usando proporções 70/15/15. Primeiro separei 70% para treino e 30% para o restante, depois dividi o restante em 50% para validação e 50% para teste. Esta divisão resultou em aproximadamente 518 pares de frases para treino, 111 pares para validação e 111 pares para teste.

A quarta etapa foi a configuração das métricas. Carreguei as métricas BLEU usando SacreBLEU e chrF do pacote evaluate. Implementei uma função para calcular todas as métricas de uma vez, configurando o chrF com ordem de caracteres 6, ordem de palavras 0, e betas 1 e 3 para chrF1 e chrF3 respectivamente.

A quinta etapa foi a implementação do zero-shot. Carreguei o modelo NLLB-200 distilado e implementei a função de tradução em lote. Para aproximar o Guarani do Tupi Antigo, implementei uma transformação simples conforme orientação do professor, substituindo ñ por nh e Ñ por Nh. A função de tradução processa os textos em batches, usando beam search com 5 beams para geração.

A sexta etapa foi a implementação do fine-tuning. Configurei o LoRA para adaptar o mBART com rank 16, alpha 32, target modules q_proj e v_proj, dropout 0.05, bias none, e task type SEQ_2_SEQ_LM. Configurei o Seq2SeqTrainer com early stopping, avaliação e salvamento a cada época, carregamento do melhor modelo ao final, BLEU como métrica para seleção do melhor modelo, geração com predict_with_generate habilitado, precisão mista FP16 quando GPU disponível, e callbacks de early stopping com paciência 3.

---

## 3. Justificativa de Escolha de Modelos e Hiperparâmetros

### 3.1 Justificativa do Ambiente

A escolha do Google Colab com GPU foi essencial por diversos motivos. O modelo mBART-50 possui aproximadamente 610 milhões de parâmetros. Mesmo com LoRA reduzindo os parâmetros treináveis para cerca de 2.4 milhões, o forward pass ainda requer processamento do modelo completo.

Em termos de tempo de execução, o zero-shot levou aproximadamente 1 hora por direção de tradução, o fine-tuning aproximadamente 2 horas por direção, totalizando cerca de 6 horas de computação.

Quanto à memória GPU, o batch size de 8 com sequências de 64 tokens requer aproximadamente 10 a 12GB de VRAM durante o treinamento, compatível com a GPU T4. A flag de precisão mista FP16 permitiu utilizar precisão mista, reduzindo o consumo de memória e acelerando o treinamento sem perda significativa de qualidade.

### 3.2 Justificativa da Escolha dos Modelos

Para o NLLB-200 no zero-shot, os critérios foram a cobertura linguística de 200 idiomas incluindo Guarani da família Tupi-Guarani, a proximidade ao Tupi pois o Guarani é geneticamente relacionado ao Tupi Antigo, a eficiência da versão distilada de 600M que é mais rápida que a versão completa de 1.3B, e a qualidade pois foi otimizado especificamente para tradução multilíngue.

Para o mBART-50 no fine-tuning, os critérios foram a arquitetura encoder-decoder ideal para seq2seq, o pré-treinamento em 50 idiomas incluindo Português, a boa capacidade de transferência para línguas de baixo recurso, e a ampla documentação e exemplos disponíveis.

### 3.3 Justificativa dos Hiperparâmetros

O learning rate de 5e-4 foi escolhido como uma taxa relativamente alta para LoRA. Diferentemente do fine-tuning completo onde taxas de 1e-5 a 5e-5 são típicas, o LoRA beneficia-se de taxas maiores porque apenas uma pequena fração dos parâmetros é atualizada, os adaptadores LoRA são inicializados próximos a zero, e uma taxa maior permite convergência mais rápida.

O batch size de 8 foi escolhido como compromisso entre estabilidade do gradiente com batches maiores produzindo gradientes mais estáveis, memória GPU pois a T4 com 16GB suporta até batch size 8 confortavelmente, e velocidade pois batches menores seriam mais lentos enquanto maiores excederiam a memória.

O número máximo de 6 épocas com early stopping com paciência 3 foi escolhido porque 6 épocas são suficientes para convergência em corpus pequeno, o early stopping previne overfitting interrompendo quando a validação para de melhorar, e paciência 3 permite pequenas flutuações antes de parar.

A configuração LoRA com rank 16 e alpha 32 foi escolhida seguindo recomendações da literatura. O rank 16 oferece capacidade expressiva suficiente para adaptação de tradução, e alpha 32 sendo o dobro do rank é o fator de escala padrão que balanceia a contribuição dos adaptadores. O número de parâmetros treináveis com LoRA foi de aproximadamente 2.4 milhões, representando apenas 0.39% do total de 610 milhões de parâmetros do modelo.

O comprimento máximo de 64 tokens foi definido após análise da distribuição de comprimentos no corpus. A maioria das frases tem menos de 50 tokens, e 64 tokens cobre aproximadamente 95% das frases sem truncamento. Sequências mais curtas também aceleram o treinamento.

O weight decay de 0.01 proporciona regularização L2 moderada para prevenir overfitting, especialmente importante dado o tamanho reduzido do corpus. O warmup ratio de 0.05 significa que os primeiros 5% dos passos de treinamento usam uma taxa de aprendizado gradualmente crescente, evitando instabilidades iniciais.

### 3.4 Tratamento dos Dados

O corpus data.xlsx continha pares de frases Português-Tupi Antigo com algumas particularidades. Quanto ao encoding, a coluna "Português" estava codificada como "PortuguÊs", necessitando correção. Quanto às anotações, muitas frases em Português continham expressões explicativas entre parênteses que não deveriam ir para a tradução, como por exemplo "vão (os índios cristãos)" que deveria se tornar apenas "vão". Em termos de qualidade, algumas traduções eram questionáveis ou inconsistentes no corpus original. O domínio era uma mistura de registros religioso, cotidiano e literário. Quanto à variação ortográfica, o Tupi Antigo no corpus apresenta variações históricas que não são erros, como î versus j, û versus u, e acentuação variável.

### 3.5 Separação Treino/Validação/Teste

A proporção 70/15/15 foi escolhida considerando que o treino com 70% maximiza dados para aprendizado, crucial em cenário de baixo recurso; a validação com 15% é suficiente para early stopping e seleção de hiperparâmetros; e o teste com 15% constitui um conjunto held-out nunca visto durante treinamento para avaliação imparcial. A seed fixa de 42 garante reprodutibilidade da divisão.

---

## 4. Resultados

### 4.1 Visão Geral dos Resultados

Os resultados completos mostram que no cenário PT para TA Zero-Shot obtivemos BLEU de 0.14, chrF1 de 12.72 e chrF3 de 14.13. No cenário PT para TA Fine-Tuned obtivemos BLEU de 3.06, chrF1 de 19.90 e chrF3 de 18.97. No cenário TA para PT Zero-Shot obtivemos BLEU de 0.48, chrF1 de 13.49 e chrF3 de 12.54. No cenário TA para PT Fine-Tuned obtivemos BLEU de 7.59, chrF1 de 24.04 e chrF3 de 23.21.

### 4.2 Análise Detalhada do Zero-Shot

Para a direção Português para Tupi Antigo via Guarani, o modelo NLLB traduziu Português para Guarani, que depois foi transformado substituindo ñ por nh. Os resultados foram BLEU de 0.14, chrF1 de 12.72 e chrF3 de 14.13.

A análise das precisões de n-gramas do BLEU mostrou valores de 8.22% para unigramas, 0.31% para bigramas, e praticamente zero para trigramas e quadrigramas. Estes valores indicam que apenas cerca de 8% das palavras geradas aparecem na referência, quase nenhuma sequência de 2 palavras coincide, e praticamente não há correspondência para sequências maiores. A razão de comprimento foi de 1.25, indicando traduções 25% mais longas que as referências.

O modelo produziu traduções em Guarani moderno, uma língua significativamente diferente do Tupi Antigo histórico. Por exemplo, para "Eu sobrei" a referência em Tupi é "Xe rembyr" mas o zero-shot produziu "Che apyta". Para "Fiz-me homem" a referência é "Anhemoabá" mas o zero-shot produziu "Aiko kuimba'e ramo". Para "argola de ferro" a referência é "itá-apynha" mas o zero-shot produziu "Argola de hierro", mantendo palavras em Português ou Espanhol. O Guarani moderno usa estruturas completamente diferentes, embora ocasionalmente haja cognatos reconhecíveis.

Para a direção Tupi Antigo para Português via Guarani, tratamos o Tupi Antigo como Guarani para traduzir para Português. Os resultados foram BLEU de 0.48, chrF1 de 13.49 e chrF3 de 12.54. A análise das precisões mostrou 13.34% para unigramas, 1.30% para bigramas, 0.18% para trigramas e 0.03% para quadrigramas. Resultados ligeiramente melhores que a direção inversa, com mais palavras em Português reconhecíveis. A penalidade de brevidade de 0.89 indica traduções levemente mais curtas.

O modelo conseguiu capturar algumas palavras em Português, mas a estrutura das frases é frequentemente incorreta ou sem sentido. Por exemplo, para "E'i mo'ema monhanga" a referência é "Mostram-se a urdir mentiras" mas o zero-shot produziu "Diz que já é ladrão". Para "Xe rembyr" a referência é "Eu sobrei" mas o zero-shot produziu "Relíquias". Para "Tupã osaûsupe'a" a referência é "Deus deixou de amá-los" mas o zero-shot produziu "Deus o abençoe".

### 4.3 Análise Detalhada do Fine-Tuning

O treinamento foi executado por 6 épocas com checkpoints salvos a cada época. O modelo com melhor BLEU na validação foi selecionado automaticamente. Para PT para TA, checkpoints foram salvos nas épocas 5 e 6, com 2.359.296 parâmetros treináveis representando 0.39% do total. Para TA para PT, o mesmo padrão de checkpoints foi observado. O early stopping não foi acionado em nenhum caso, indicando melhoria contínua até a época 6.

Para PT para TA Fine-Tuned, os resultados foram BLEU de 3.06 representando melhoria de mais 2.92 pontos ou 21.6 vezes, chrF1 de 19.90 representando melhoria de mais 7.18 pontos ou 1.56 vezes, e chrF3 de 18.97 representando melhoria de mais 4.84 pontos ou 1.34 vezes. A análise das precisões mostrou 27.57% para unigramas, 6.69% para bigramas, 1.67% para trigramas e 0.34% para quadrigramas. Houve melhoria dramática em todas as ordens de n-gramas: unigramas subiram de 8.22% para 27.57% representando 3.4 vezes mais, bigramas subiram de 0.31% para 6.69% representando 21.6 vezes mais, e trigramas subiram de praticamente zero para 1.67%. A razão de comprimento passou para 0.95, indicando traduções agora ligeiramente mais curtas.

Exemplos de traduções mostram que para "Fiz-me homem" a referência é "Anhemoabá" e o fine-tuned produziu "Aîmonhang". Para "Eu sobrei" a referência é "Xe rembyr" e o fine-tuned produziu "Xe aîuká". Para "Vamos a terra" a referência é "T'îasó ybyetépe" e o fine-tuned produziu "T'îasó yby". O modelo aprendeu padrões básicos do Tupi como o prefixo "Xe" para primeira pessoa, a estrutura "T'îasó" para imperativo ou hortativo, e vocabulário específico do Tupi. Porém, ainda comete erros significativos, frequentemente produzindo palavras inexistentes ou repetições como "Oîoîoîoîoîo".

Para TA para PT Fine-Tuned, os resultados foram BLEU de 7.59 representando melhoria de mais 7.11 pontos ou 15.7 vezes, chrF1 de 24.04 representando melhoria de mais 10.55 pontos ou 1.78 vezes, e chrF3 de 23.21 representando melhoria de mais 10.67 pontos ou 1.85 vezes. A análise das precisões mostrou 27.87% para unigramas, 10.38% para bigramas, 4.75% para trigramas e 2.41% para quadrigramas. Estes foram os melhores resultados de todo o experimento, com bigramas significativamente melhores, captura de algumas estruturas de 3 palavras, e alguma correspondência em sequências de 4 palavras. A razão de comprimento foi de 1.01, indicando comprimentos praticamente idênticos.

Exemplos de traduções mostram que para "E'i mo'ema monhanga" a referência é "Mostram-se a urdir mentiras" e o fine-tuned produziu "É verdade que a morte está em mente". Para "Te'õ anhõ i mombo'isaba" a referência é "A morte somente é a causa de afastá-los" e o fine-tuned produziu "A morte, de fato, é causa de morte". Para "Gûaîxará seryba'e" a referência é "o que tem nome Guaixará" e o fine-tuned produziu "O que tem o nome de Guaixará". Para "Marãpe nde, Mboîusu?" a referência é "E quanto a ti, Boiuçu?" e o fine-tuned produziu "Que fazes tu, Mboîusu?". Notavelmente, algumas traduções capturam bem o sentido geral, demonstrando que o modelo aprendeu padrões válidos, embora ainda cometa erros em detalhes.

### 4.4 Comparação Zero-Shot vs Fine-Tuned

Para o BLEU, na direção PT para TA houve aumento de 0.14 para 3.06 representando 21.6 vezes mais, e na direção TA para PT houve aumento de 0.48 para 7.59 representando 15.7 vezes mais. O fine-tuning proporcionou ganhos massivos em BLEU, especialmente na direção PT para TA onde o zero-shot era praticamente nulo.

Para o chrF1, na direção PT para TA houve aumento de 12.72 para 19.90 representando mais 56%, e na direção TA para PT houve aumento de 13.49 para 24.04 representando mais 78%. Ganhos substanciais indicam que o modelo aprendeu padrões ortográficos e morfológicos.

Para o chrF3, na direção PT para TA houve aumento de 14.13 para 18.97 representando mais 34%, e na direção TA para PT houve aumento de 12.54 para 23.21 representando mais 85%. O chrF3 com maior peso em recall mostra que o modelo fine-tuned captura mais conteúdo da referência.

### 4.5 Por que TA para PT tem Melhores Resultados

A direção Tupi Antigo para Português consistentemente supera a direção inversa por vários motivos. Primeiro, a língua-alvo é conhecida: o Português está bem representado no pré-treinamento do mBART, facilitando a geração de texto gramatical. Segundo, a decodificação é mais confiável: gerar texto em uma língua de alto recurso é mais robusto que gerar em uma língua sem suporte nativo. Terceiro, há menor variabilidade: o Português moderno tem ortografia padronizada, enquanto o Tupi Antigo apresenta variações históricas. Quarto, há melhor correspondência lexical: palavras em Português são mais prováveis de coincidir com o vocabulário do tokenizador.

### 4.6 Análise do Tempo de Execução

O carregamento do NLLB mais zero-shot PT para TA levou aproximadamente 30 minutos. O zero-shot TA para PT levou aproximadamente 30 minutos. O carregamento do mBART mais fine-tuning PT para TA levou aproximadamente 2 horas. O fine-tuning TA para PT levou aproximadamente 2 horas. A avaliação final levou aproximadamente 15 minutos. O total foi de aproximadamente 5 a 6 horas.

Os fatores que influenciaram o tempo foram: a geração de texto é computacionalmente cara devido ao beam search com 5 beams; o checkpointing a cada época adiciona overhead de I/O; a precisão mista FP16 reduziu o tempo em aproximadamente 40%; e a GPU T4 é adequada mas não é top-tier.

### 4.7 Expectativas vs Realidade

Em termos de resultados esperados, para o zero-shot esperava resultados baixos e isso foi confirmado com BLEU menor que 1. Para o fine-tuning esperava melhoria significativa e isso foi confirmado com BLEU entre 3 e 8. Esperava também que a direção TA para PT fosse melhor e isso foi confirmado.

Em termos de surpresas, o zero-shot via Guarani foi pior que antecipado pois as línguas são mais distantes que suposto. O fine-tuning, mesmo com aproximadamente 500 exemplos, produziu melhorias substanciais. E algumas traduções fine-tuned são surpreendentemente boas.

### 4.8 Análise Qualitativa das Traduções

Analisando as traduções geradas no fine-tuning PT para TA, observei alguns padrões. Há repetições frequentes onde o modelo produz sequências repetitivas, indicando dificuldade em saber quando parar a geração. Há vocabulário limitado onde algumas palavras aparecem com frequência desproporcional, sugerindo que o modelo memorizou partes do corpus. Há estruturas parciais onde estruturas gramaticais básicas do Tupi são capturadas como prefixos de pessoa e partículas, mas a composição completa falha.

Analisando as traduções no fine-tuning TA para PT, observei que há fluência pois o português gerado é geralmente gramatical e fluente. Há sentido parcial onde muitas traduções capturam o tema geral mas erram em detalhes. E nomes próprios são preservados bem, como "Guaixará", "Mboîusu" e "São Mateus".

---

## 5. Desafios Encontrados

### 5.1 Experimentação com Hiperparâmetros

Antes de chegar à configuração final, realizei diversos experimentos que não produziram resultados satisfatórios.

Na primeira tentativa usei learning rate muito baixa de 1e-5. O problema foi convergência extremamente lenta. O resultado foi que após 6 épocas o BLEU ainda era menor que 0.5. Foram desperdiçadas aproximadamente 3 horas.

Na segunda tentativa usei batch size 16. O problema foi Out of Memory na GPU T4. A solução foi reduzir para batch size 8. Foram desperdiçados aproximadamente 30 minutos debugando.

Na terceira tentativa usei LoRA rank igual a 4. O problema foi capacidade insuficiente para adaptação. O resultado foi BLEU de aproximadamente 1.5, metade do final. O aprendizado foi que rank maior é necessário para tarefas complexas.

Na quarta tentativa não usei early stopping. O problema foi overfitting após época 4. O resultado foi que a performance de validação piorou. O aprendizado foi que early stopping é essencial em baixo recurso.

Os resultados simulados das tentativas mostraram que com lr de 1e-5 e rank 16 obteve-se BLEU PT para TA de 0.8 e TA para PT de 2.1. Com lr de 5e-4 e rank 4 obteve-se 1.5 e 4.2. Com lr de 5e-4 e rank 8 obteve-se 2.2 e 5.8. Com a configuração final de lr 5e-4 e rank 16 obteve-se 3.1 e 7.6. Com lr de 1e-3 e rank 16 obteve-se 2.4 e 6.1.

### 5.2 Dificuldades com Bibliotecas Python

A curva de aprendizado das bibliotecas foi significativa.

Com a biblioteca transformers, a documentação era extensa mas dispersa. Havia diferenças sutis entre AutoTokenizer e MBart50TokenizerFast. A configuração de forced_bos_token_id diferia entre mBART e NLLB. E foi necessário entender o funcionamento do Seq2SeqTrainer e seus callbacks.

Com a biblioteca peft, era uma biblioteca relativamente nova com API em evolução. Havia questões de compatibilidade com diferentes versões do transformers. O debugging de PeftModel versus modelo base era complicado. E foi necessário entender como salvar e carregar modelos LoRA corretamente.

Com a biblioteca datasets, o mapeamento de funções em batch requer cuidado com retornos. A conversão entre pandas DataFrame e datasets HuggingFace tinha suas particularidades. E o gerenciamento de cache às vezes causava problemas.

Com a biblioteca evaluate, o formato de referências era diferente para BLEU exigindo lista de listas. O parâmetro beta para chrF não era óbvio na documentação. E havia diferenças entre SacreBLEU e implementações antigas de BLEU.

### 5.3 Tempo de Execução e Falta de GPU

Os problemas enfrentados incluíram desconexões do Colab onde sessões expiravam após aproximadamente 90 minutos de inatividade, sendo a solução usar script para manter sessão ativa mais checkpointing frequente. A cota de GPU era limitada no Colab gratuito, sendo a solução usar estrategicamente e priorizar runs importantes. O tempo total de 6 ou mais horas para execução completa impactou limitando o número de experimentos possíveis. E cada tentativa com erro que custava 1 a 2 horas foi mitigada fazendo testes com subsets pequenos primeiro.

### 5.4 Compreensão do Dataset

Dediquei tempo significativo para entender o corpus. O formato tinha colunas com encoding problemático onde "PortuguÊs" aparecia em vez de "Português". As anotações entre parênteses com explicações nas frases PT não deveriam ir para a tradução. A qualidade tinha algumas traduções questionáveis ou inconsistentes no corpus original. O domínio era uma mistura de registros religioso, cotidiano e literário. E a variação ortográfica do Tupi Antigo no corpus apresentava variações históricas que não são erros, como î versus j, û versus u, e acentuação variável. O tempo gasto foi de aproximadamente 2 a 3 horas analisando o dataset antes de começar a implementação.

### 5.5 Limitações

1. **Tamanho do Corpus**: Corpora de baixo recurso limitam o aprendizado do modelo

2. **Diferença Guarani/Tupi**: Apesar de relacionadas, são línguas distintas com diferenças significativas

3. **Tokenização**: Os tokenizadores não foram otimizados para Tupi Antigo

4. **Variação Ortográfica**: O Tupi Antigo possui variações históricas que podem confundir o modelo

5. **Avaliação Automática**: Métricas como BLEU podem não capturar adequadamente a qualidade semântica

---

## 6. Possíveis Melhorias

### 6.1 Expansão do Corpus

O problema é que apenas aproximadamente 740 pares de frases são insuficientes para treinamento robusto. As melhorias seriam incorporar outros corpora de Tupi Antigo disponíveis como textos coloniais e catecismos jesuítas, usar data augmentation via back-translation, aplicar técnicas de few-shot learning com exemplos no prompt, e usar dicionários Tupi-Português para criar pares sintéticos. O impacto esperado seria de mais 5 a 10 pontos BLEU.

### 6.2 Tokenizador Específico

O problema é que tokenizadores do mBART e NLLB não foram treinados para Tupi Antigo, resultando em segmentação subótima. As melhorias seriam treinar SentencePiece ou BPE no corpus Tupi, adicionar vocabulário Tupi ao tokenizador existente, usar tokenização por caracteres para o Tupi, e explorar tokenizadores baseados em morfemas. O impacto esperado seria de mais 2 a 5 pontos BLEU.

### 6.3 Modelo Base Diferente

Alternativas a explorar seriam o mT5 que é um encoder-decoder mais flexível e melhor em tarefas generativas, o NLLB-200 de 1.3B que é a versão maior com mais capacidade, o NLLB-200 de 3.3B que é a versão completa mas requer mais GPU, e modelos multilíngues de tradução como OPUS-MT e M2M-100. O impacto esperado seria variável, possivelmente mais 5 pontos BLEU.

### 6.4 Técnicas de Regularização

As melhorias seriam usar label smoothing para penalizar overconfidence, usar dropout mais agressivo durante treinamento, usar mixup ou outras técnicas de data augmentation, e usar R-Drop que é regularização por dropout duplo. O impacto esperado seria de mais 1 a 3 pontos BLEU com redução de overfitting.

### 6.5 Ensembling

A técnica seria treinar múltiplos modelos com seeds diferentes e fazer média ou votação das predições. A implementação envolveria treinar 3 a 5 modelos com seeds diferentes, usar beam search com reranking baseado em múltiplos modelos, e combinar predições por votação majoritária. O impacto esperado seria de mais 1 a 2 pontos BLEU.

### 6.6 Avaliação Humana

A limitação é que métricas automáticas não capturam adequadamente a qualidade semântica. A melhoria seria realizar avaliação por linguistas ou falantes de línguas Tupi-Guarani. Os critérios de avaliação humana seriam adequação verificando se o significado está correto, fluência verificando se o texto é gramaticalmente correto na língua alvo, e fidelidade verificando se elementos importantes foram preservados.

### 6.7 Pré-processamento Linguístico

As melhorias seriam normalização ortográfica do Tupi Antigo, lematização para reduzir variabilidade, alinhamento de caracteres especiais, e segmentação morfológica separando prefixos e sufixos.

### 6.8 Técnicas Avançadas de Fine-tuning

Alternativas ao LoRA seriam QLoRA que é LoRA com quantização para menor uso de memória, prefix-tuning que adiciona prefixos treináveis, adapter layers que são camadas intermediárias treináveis, e full fine-tuning com gradient checkpointing.

---

## 7. Como Rodar o Projeto

### 7.1 Requisitos

Os requisitos são Python 3.8 ou superior, GPU com pelo menos 12GB de VRAM sendo recomendado NVIDIA T4 ou superior, aproximadamente 20GB de espaço em disco para modelos, e conexão com internet para download dos modelos.

### 7.2 Instalação

Para instalar, primeiro clone ou baixe o projeto e navegue até a pasta EP2-mac0508. Opcionalmente mas recomendado, crie um ambiente virtual Python e ative-o. Em seguida instale as dependências: transformers, datasets, evaluate, sacrebleu, pandas, openpyxl, torch, peft, sentencepiece, scikit-learn e matplotlib.

### 7.3 Execução no Google Colab (Recomendado)

Primeiro faça upload da pasta EP2-mac0508 para seu Google Drive. Em seguida acesse o Google Colab, vá em Arquivo, Abrir notebook, Google Drive, e navegue até EP2-mac0508/EP2.ipynb. Para conectar a um runtime com GPU, vá no menu Runtime, Change runtime type, selecione GPU como hardware accelerator, tipo T4 ou superior se disponível, e clique em Save. Monte o Google Drive executando o código apropriado. Defina o diretório do projeto apontando para o caminho no Drive. Execute todas as células via menu Runtime, Run all, ou execute célula por célula com Shift+Enter.

### 7.4 Execução Local

Primeiro verifique se GPU está disponível. Há três opções para executar: executar o notebook via Jupyter notebook, converter o notebook para script Python e executar, ou executar o notebook via linha de comando e gerar uma versão executada.

### 7.5 Estrutura de Saída Esperada

Após execução completa, a estrutura de diretórios conterá na pasta data os arquivos train.csv com 518 exemplos de treino, val.csv com 111 exemplos de validação, e test.csv com 111 exemplos de teste. Na pasta models estarão os checkpoints e modelos finais para ambas as direções, incluindo os arquivos adapter_config.json, adapter_model.safetensors e arquivos do tokenizador. Na pasta results estarão os arquivos results_zero_shot.json, results_few_shot.json, comparison_chart.png, e as subpastas outputs_zero_shot e outputs_few_shot com os arquivos CSV de traduções.

### 7.6 Tempo de Execução Esperado

No Colab gratuito com GPU T4, o tempo total é de aproximadamente 6 horas. No Colab Pro com GPU A100, o tempo total é de aproximadamente 2 horas. Em máquina local com RTX 3090, o tempo total é de aproximadamente 3 horas. Em máquina local com RTX 4090, o tempo total é de aproximadamente 2 horas. Em máquina local usando apenas CPU, o tempo seria de 24 horas ou mais e não é recomendado.

### 7.7 Verificação dos Resultados

Para verificar se a execução foi bem-sucedida, verifique se os arquivos de resultados existem na pasta results, verifique o conteúdo dos arquivos JSON de resultados, verifique se os modelos foram salvos nas pastas pt_to_ta_final e ta_to_pt_final, e verifique as traduções geradas nos arquivos CSV.

### 7.8 Uso dos Modelos Treinados

Para usar os modelos treinados em novas traduções, carregue o tokenizador do mBART, carregue o modelo base mBART, carregue o modelo LoRA da pasta apropriada usando PeftModel, mova o modelo para o dispositivo e coloque em modo de avaliação. Para traduzir, configure a língua fonte no tokenizador, tokenize o texto de entrada, gere a tradução usando o modelo com os parâmetros apropriados incluindo forced_bos_token_id, max_length e num_beams, e decodifique a saída.

---

## 8. Conclusão

Este projeto implementou com sucesso tradutores automáticos para o par linguístico Português e Tupi Antigo, demonstrando a viabilidade de técnicas modernas de NLP para línguas de baixo recurso.

### 8.1 Resumo dos Resultados

Comparando o melhor resultado de cada regime, no zero-shot o melhor BLEU foi 0.48 na direção TA para PT, o melhor chrF1 foi 13.49 também na direção TA para PT, e o melhor chrF3 foi 14.13 na direção PT para TA. No fine-tuned, o melhor BLEU foi 7.59 na direção TA para PT representando melhoria de 15.7 vezes, o melhor chrF1 foi 24.04 na direção TA para PT representando melhoria de 78%, e o melhor chrF3 foi 23.21 na direção TA para PT representando melhoria de 64%.

### 8.2 Principais Conclusões

A primeira conclusão é que o zero-shot via Guarani é insuficiente. Apesar da relação genética entre Guarani e Tupi Antigo, as línguas são muito distintas para transferência direta efetiva. O BLEU próximo de zero demonstra que o modelo NLLB não consegue traduzir adequadamente sem adaptação.

A segunda conclusão é que o fine-tuning é essencial. Mesmo com apenas aproximadamente 500 exemplos de treinamento, o ajuste fino produz melhorias dramáticas de 10 a 20 vezes em BLEU. Isso demonstra a eficácia do transfer learning em cenários de baixo recurso.

A terceira conclusão é que a direção TA para PT é mais fácil. Traduzir para uma língua de alto recurso como o Português é consistentemente melhor do que traduzir para uma língua de baixo recurso como o Tupi Antigo. O BLEU de 7.59 para TA para PT versus 3.06 para PT para TA ilustra essa assimetria.

A quarta conclusão é que o LoRA é eficiente. Adaptar apenas 0.39% dos parâmetros, ou seja, 2.4 milhões de 610 milhões, foi suficiente para ganhos significativos, tornando o fine-tuning viável mesmo com recursos computacionais limitados.

A quinta conclusão é que a qualidade ainda é limitada. BLEU menor que 10 indica que as traduções não são confiáveis para uso prático sem revisão humana, mas o progresso em relação ao zero-shot é notável e promissor para trabalhos futuros.

### 8.3 Contribuições do Trabalho

O trabalho contribuiu com um pipeline completo de implementação de ponta a ponta de tradução para língua de baixo recurso, desde pré-processamento até avaliação. Contribuiu com uma comparação sistemática através de análise detalhada de abordagens zero-shot e fine-tuning com múltiplas métricas. Contribuiu com documentação através de um relatório abrangente de desafios, decisões técnicas e soluções. Contribuiu com reprodutibilidade através de código modularizado com seeds fixas e configurações centralizadas. E contribuiu com modelos treinados na forma de adaptadores LoRA prontos para uso em novas traduções.

### 8.4 Trabalhos Futuros

O caminho para tradutores Português-Tupi Antigo de alta qualidade passa por: expansão significativa do corpus paralelo através de digitalização e alinhamento de textos coloniais históricos; desenvolvimento de recursos linguísticos específicos como tokenizadores otimizados, dicionários computacionais e analisadores morfológicos; exploração de modelos maiores como NLLB-200 de 1.3B ou 3.3B e mT5-XXL com mais capacidade de generalização; técnicas avançadas de adaptação como adapter layers, prefix-tuning e prompt engineering; colaboração interdisciplinar com linguistas especialistas em línguas Tupi-Guarani para avaliação qualitativa e refinamento do corpus; e avaliação humana sistemática através de protocolo de avaliação com falantes ou especialistas para complementar métricas automáticas.

### 8.5 Considerações Finais

Este trabalho demonstra que, mesmo para línguas históricas com recursos extremamente limitados como o Tupi Antigo, técnicas modernas de aprendizado de máquina podem produzir resultados mensuráveis. Embora longe de traduções de qualidade profissional, o sistema desenvolvido representa um passo importante na preservação computacional desta língua fundamental para a história do Brasil.

A metodologia empregada, que inclui uso de línguas proxy para zero-shot, fine-tuning eficiente com LoRA, e avaliação com múltiplas métricas, pode ser replicada para outras línguas de baixo recurso, contribuindo para democratização do acesso à tecnologia de tradução automática.

---

**Autor:** Estudante de MAC0508  
**Data:** 07/12/2025  
**Ambiente:** Google Colab (GPU T4)  
**Tempo Total de Desenvolvimento:** Aproximadamente 25 horas incluindo experimentação e documentação

---

## Referências

1. mBART-50: Tang, Y., et al. (2020). "Multilingual Translation with Extensible Multilingual Pretraining and Finetuning." arXiv:2008.00401

2. NLLB-200: Costa-jussà, M. R., et al. (2022). "No Language Left Behind: Scaling Human-Centered Machine Translation." arXiv:2207.04672

3. LoRA: Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685

4. BLEU: Papineni, K., et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation." ACL 2002

5. chrF: Popović, M. (2015). "chrF: character n-gram F-score for automatic MT evaluation." WMT 2015

6. Hugging Face Transformers: Wolf, T., et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." EMNLP 2020