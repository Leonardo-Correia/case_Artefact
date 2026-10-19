# Treinamento e orçamento de verificação

O treinamento em lote usa os dados rotulados de 2025 e todas as features atuais.
Não compara com a versão antiga nem treina incrementalmente.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python training_workflow.py
python -m unittest discover -s tests -v
```

Os quatro modelos são regressão logística, HistGradientBoosting, Random Forest e
MLP. A divisão aleatória reserva 84% para treino, 8% para validação e 8% para
teste. Scaler, encoder e imputador são ajustados apenas no treino. O imputador
usa mediana e indicadores de ausência. Os artefatos ficam em
`artifacts/training/`, incluindo os índices das divisões, metadados, modelos,
previsões e métricas de validação. Os scores não são calibrados.

O notebook carrega esses artefatos e apresenta curvas ROC e precision-recall.
Para refazer o treinamento, execute o comando acima ou `run_training()`.

## Restrição de orçamento

`threshold_for_budget(scores, verification_cost=X, budget=Y)` permite no máximo
`floor(Y/X)` verificações. Toda verificação custa X, inclusive falso positivo.
Os valores monetários aceitam até duas casas decimais. X deve ser positivo e Y
não negativo. A capacidade é calculada com aritmética decimal.

O limiar usa comparação inclusiva `score >= threshold`. Scores empatados nunca
são separados: um grupo que excederia o orçamento fica fora, podendo sobrar
verba. Limiar infinito significa nenhuma verificação. A função não usa rótulos.
A fila é recalculada para cada lote, sem atualizar o modelo. Um limiar fixo
estimado em outro lote não garante o teto de gasto em um lote futuro.

`cost_curve(y_true, scores, X)` calcula gastos, precisão e recall observados em
cada limiar distinto. A validação permite escolher o modelo que recupera mais
fraudes dentro do orçamento. O teste é usado depois dessa escolha; seus rótulos
não orientam o modelo nem o limiar. Na avaliação por segmento, o orçamento
continua global para o lote, sem ser multiplicado pelo número de segmentos.

Configure X, Y e período no notebook. O orçamento precisa corresponder ao lote:
não aplique automaticamente o orçamento de um mês inteiro a uma amostra ou a
cada segmento. Há um parâmetro separado para o mês completo de janeiro.

A restrição de gasto não representa um modelo completo de perdas: não foram
fornecidos custos de fraude perdida, bloqueio indevido ou benefício da detecção.
A feature de fraudes anteriores mantém as hipóteses do projeto original sobre
a disponibilidade dos rótulos; janeiro não contém confirmações de novas fraudes.
