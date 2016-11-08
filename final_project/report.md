# Enron Submission Free-Response Questions

### 1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

O objetivo deste projeto é identificar funcionários da Enron, chamados Pessoas de Intesse (POIs), que possam ter cometido fraude. Foram utilizas informações públicas divulgadas durante a investigação da gigante norte americana, que atuava principalmente no ramo de energia. Chamada pela revista Fortune de "America's Most Innovative Company" por seis anos consecutivos, decretou falência em 2001.

Os outilers encontrados durante a investigação foram removidos da seguinte forma: 1) removido "TOTAL", por não se tratar de uma POI, e sim de um totalizador da planilha; 2) para cada feature, remover os outliers acima do 3º quartil, desde que não estivessem relacionados a uma POI e não ultrapassando 5% dos dados; 3) para cada feature, remover dos outilers abaixo do 1º quartil, desde que não estivessem relacionados a uma POI e não ultrapassando 5% dos dados.

### 2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

As features foram selecionadas através da ferramenta SelectKBest do sklearn, que seleciona as K melhores features, utilizando, neste caso, ANOVA F-value como função de avaliação. Como as magnitudes são muito diferentes entre e-mails enviados ou recebidos e os valores monetários, todas as features foram reescaladas para valores entre 0 e 1. 

Foram criadas duas novas features: 1) receita: salário + bônus, dando um peso maior a estes ganhos do que eles teriam se estivessem isolados; 2) pares de e-mails: o menor valor entre as quantidades de e-mails enviados para um POI e e-mails recebido de um POI, que representa a troca de mensagens (envio e resposta) entre POIs. Para aumentar a diferença entre os valores mais baixos e os mais altas, as novas features foram elevadas ao quadrado. 

Os 8 maiores scores foram: exercised_stock_options: 25.38; total_stock_value: 24.75; income: 23.18; bonus: 21.32; salary: 18.86; deferred_income: 11.73; restricted_stock: 9.48. Foram escolhidos exercised_stock_options, income, deferred_income, long_term_incentive. Não foram mantidos: total_stock_value e restricted_stock por se tratarem do mesmo contexto de exercised_stock_options, e que possui um score mais alto; bonus e salary, pois income resume os dois e possui score mais alto. Nenhum informação de troca de e-mails foi utilizada. Aparentemente a máxima "siga o dinheiro" também se aplica neste caso.

### 3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

Algoritmos testados: GaussianNB, DicisionTree, AdaBoost, GaussianNB/PCA, SVC, KNeighborsClassifier e RandomForest. Em alguns casos os parâmetros foram escolhidos automaticamente através do GridSearchCV. Os 4 primeiros da lista obtiveram precision e recall maior que 0.3:

|Algoritmo                |Precision|Recall |
|-------------------------|---------|-------|
|GaussianNB               |0.50902	|0.39500|
|**GaussianNB/PCA**       |0.55469	|0.39050|
|DicisionTree             |0.30242	|0.31300|
|AdaBoost                 |0.30717	|0.31700|

### 4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

Foram realizados ajustes automáticos de parâmetros, utilizando GridSearchCV, para os algoritmos: SVC (parâmetros kernel, gamma e C), DecisionTree (parâmetros min_samples_split, max_features, min_samples_leaf, criterion, max_depth, max_leaf_nodes), KNeighbors (parâmetros metrics, weights, leaf_size).

No caso do GaussianNB foi utilizada PCA para reduzir a dimensionalidade das features. Foram testados manualmente alguns números de componentes até a escolha de n_components=3.

### 5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

Os resultados foram confrontados com K Fold Cross Validation e com as métricas precision e recall. Uma das grandes dificuldades é manter o algortimo equilibrado entre precision e recall. É comum obtermos valores altos para apenas uma destas métricas enquanto a outra despenca o seu resultado.

### 6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

As principais métricas utilizadas foram precision e recall. Recall mede o quão completos são os resultados. Quanto mais alta esta métrica maior a possibilidade de selecionar um verdadeiro positivo, ainda que falsos positivos venham a ocorrer. Quando queremos garantir o maior número de casos a investigar, esta é principal métrica a ser avaliada. Já Precision mede o quão úteis são os resultados. Através dela, vamos medir os nossos acertos, ainda que alguns resultados positivos sejam descartados. 