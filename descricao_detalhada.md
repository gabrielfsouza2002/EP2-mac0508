MAC0508 — Introdução ao Processamento de Língua Natural

EP 2– Tradução Automática de Baixo Recurso
Implementação, Treinamento e Avaliação de Modelos Baseados em LLM para Tradução

Português ↔ Tupi Antigo

Entrega improrrogável: 07/12/2025

Este EP pode ser feito em dupla

O objetivo deste exercício programa (EP) é treinar e avaliar modelos de tradução automática em um cenário de baixo recurso, utilizando um córpus paralelo Português-Tupi Antigo e abordagens baseadas em modelos de linguagem de grande porte (LLMs). Vocês deverão produzir experimentos completos nos regimes zero-shot learning e few-shot learning, em ambas as direções de tradução.

Quando se fala de tarefas de processamento de texto utilizando a tecnologia de geração de texto, existem dois modos básicos de operação. O modo zero-shot (sem dicas) indica que a tarefa tem que ser realizada apenas por solicitação, sem apresentar qualquer exemplo.

É a tarefa que o programa realiza só com o treinamento genérico a que foi submetido. Já o modo few-shot (poucas dicas) indica que alguns exemplos, em geral um número não muito grande, deverão ser fornecidos antes da realização da tarefa. Nesse caso, a gente diz que o modelo foi refinado (fine-tuned) para a realização da tarefa. Nesse exercício vamos tratar de gerar tradutores tanto no modo zero-shot quanto no modo few-shot. 1. E vamos comparar as saídas dos dois modos.

Não existem medidas automáticas muito boas para se avaliar uma tradução. Mas algumas medidas são usadas tradicionalmente, e devem servir para comparar a qualidade de 2 métodos diferentes. Neste exercício vamos utilizar as seguintes medidas:

• BLEU (Bilingual Evaluation Understudy) avalia a qualidade de textos traduzidos automaticamente de um idioma para outro. As pontuações são calculadas para segmentos traduzidos individualmente — geralmente frases — comparando seus n-gramas com um conjunto de traduções de referência de boa qualidade (padrão ouro). Essas pontuações são então calculadas em média para todo o córpus, resultando num número entre 0 e 1 que indica o quão semelhante o texto candidato é aos textos de referência, com valores mais próximos de 1 representando textos mais semelhantes. 2

• chrF (CHaRacter-level F-score) é uma métrica para avaliação de tradução automática que calcula a similaridade entre uma tradução automática e uma tradução de referência usando n-gramas de caracteres, e não n-gramas de palavras. Métricas baseadas em n-gramas de palavras são especialmente problemáticas para línguas com morfologia complexa.

1É importante dizer que recentemente few-shot se tornou popular com os modelos decoder tipo GPT onde ao invés de realizar o fine-tuning, o modelo recebe exemplos diretamente no prompt, isto é, não há propagação de gradientes e o modelo aprende com base no contexto (In-Context learning)
