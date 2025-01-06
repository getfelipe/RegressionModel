
# Previsão de Preços de Casas com Regressão Linear 

Aplicando os processos de um projeto de Data Science, desde da análise exploratória até o deploy do aplicativo, que tem como objetivo prever os preços dos imóveis localizados na região da Califórnia, os condados, para compra através da idade do móvel, região e renda do possível proprietário.

Consulte aqui: https://app-california-properties-price.onrender.com/


## Lessons Learned

Criação de Pipeline para executar o modelo, utilizando transformações na coluna target, como PowerTransformer e para as variáveis preditoras RobustScaler, além disso, aplicado várias operações com Pandas.DataFrame, como o pd.cut. Também, feito a comparação de diferentes modelos lineares com GridSearch, adicionando Polinômios, modelo Ridge, Lasso, em que o Ridge teve uma performance melhor, com métricas R2 e RMSE.


## Screenshots

![App Screenshot](https://github.com/getfelipe/RegressionModel-PropertyPrice/blob/master/california-properties.png)

## Feedback

Como a base que foi utilizada para treinar dados não possuía muitos dados limpos, algumas previsões dos preços de imóveis não são muito precisas, e também visto que foi aplicado o RobustScaler, valores muito grande da renda ou muito pequeno acaba não performando muito bem, pois o RobustScaler pega valores que estão mais centralizados.

