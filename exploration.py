import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('housing.csv')
df.info()

# Com a descrição é possível nota que os valores máximos do total de quartos, população, domícilios, receita média e o valor médio da casa
# possuem valores muito acimas do 3º quartil, isso indica alguns valores extremos.
df.describe()
df.describe(exclude='number')


# É possível observar que os valores da nossa variável alvo "median_value_house" possui ṕontuações de dispersões retas no limite do gráfico, 
# isso indica que valores que ultrapassam o limite foram arrendodados para um único valor.
# Além disso, alguns outliers atrapalham na visualização das relações.
#sns.pairplot(df, diag_kind='kde', plot_kws=dict(alpha=0.2))

# O parâmetro skew nas colunas indica a assimetria da distribuição dos valores em relação à média. 
# Valores positivos de skew sugerem que a distribuição possui uma cauda mais longa à direita, indicando a presença de valores maiores que a média. 
# Colunas como total_rooms, total_bedrooms, population, households, median_income e median_house_value apresentam assimetrias positivas, sugerindo a existência de alguns valores grandes que estendem a cauda para a direita.
df.select_dtypes('number').skew()

#O parâmetro kurtosis mede se a distribuição é mais pontuda (leptocúrtica) ou achatada (platicúrtica) em comparação com a distribuição normal. 
# Valores muito positivos indicam uma distribuição com um pico central alto e caudas mais longas, sugerindo alta concentração de valores próximos à média, mas também a presença de valores extremos nas caudas.
df.select_dtypes('number').kurtosis()

# Verificando se possui registros duplicados.
df[df.duplicated()]


# Verifica se existem valores nulos.
df[df.isnull().any(axis=1)]

# Verificando se os registros com certos valores nulos possuem alguma relação.
df[df.isnull().any(axis=1)].describe()

# Verifica a quantidade de registros categóricos na variável ocean_proximity. 
# Dessa forma, categorias com poucos registros podem ser identificadas, já que ao aplicar um modelo de aprendizado, classes com amostras muito pequenas podem levar a problemas de generalização ou convergência, especialmente em algoritmos mais sensíveis a distribuições desbalanceadas. 
# Caso necessário, categorias com poucos dados podem ser descartadas, combinadas ou tratadas com técnicas adequadas.
df['ocean_proximity'].value_counts()


df[df['ocean_proximity'] == 'ISLAND'].describe()


# Aplicando o gráfico boxplot para analisar os percentis das colunas numéricas, é possível visualizar os outliers existentes nas colunas total_rooms, total_bedrooms, popylation, households, median_income e median_house_value.
# Isso traduz a grande quantidade de valores menores.
fig, axis = plt.subplots(3, 3)
for ax, col in zip(axis.flatten(), df.select_dtypes('number').columns):
    sns.boxplot(data=df, x=col, ax=ax, showmeans=True)

plt.tight_layout()
plt.show()

# A seguinte função 'np.triu' gera a máscara para o gráfico de calor, calculando somente a parte triangular superior da matriz, o que evita duplicidade entre as relações das colunas numéricas.
mask = np.triu(np.ones_like(df.select_dtypes('number').corr(), dtype=bool))

# No gráfico, nota-se que a única coluna que apresenta uma correlação mais forte com a nossa variável alvo é a median_income. 
# Isso pode ser um problema ao aplicarmos o modelo de Machine Learning, pois o modelo pode se tornar excessivamente dependente dessa variável, resultando em um viés ou em um modelo menos generalizável.
# Também, existem correlações bastante próximas de 1, como a households x total_bedrooms, o que traduz a ideia que as duas variáveis estão fazendo a mesma coisa. Nesse sentido, pode-se descartar uma delas.
# Por outro lado, é possível criar novas observações com o cruzamento dos dados, é o que chamamos de Engenharia de Atributos, para criar uma relação que o modelo possa entender melhor e fique mais clara. 
# Isso pode ser feito através da estratificação, dividindo uma coluna numérica em classes ou através de algum cálculo matemático, como a razão, multiplicação, etc.
fig, ax = plt.subplots()
sns.heatmap(
    df.select_dtypes('number').corr(),
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    ax=ax
)
plt.show()


# Logo, podemos criar derivadas das variáveis, como: classes em median_income, cômodos por domicílio, pessoas por domicílio, quarto por cômodos.
df['median_income_cat'] = pd.cut(
    df['median_income'],
    bins=[0, 1.5, 3, 4.5, 6, np.inf],
    labels=[1, 2, 3, 4, 5]
)
df['median_income_cat'].value_counts().sort_index().plot(kind='bar')

df['rooms_per_household'] = df['total_rooms'] / df['households']

df['population_per_household'] = df['population'] / df['households']

df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']

df['housing_median_age_cat'] = pd.cut(
    df['housing_median_age'],
    bins=[1,10, 20,30, 40, 50, np.inf],
    labels=[1, 2, 3, 4, 5, 6]
)
df['housing_median_age_cat'].value_counts().sort_index().plot(kind='bar')

df.describe()


# Aplicando o heatmap novamente:

mask = np.triu(np.ones_like(df.select_dtypes('number').corr(), dtype=bool))
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(
    df.select_dtypes('number').corr(),
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    ax=ax
)
plt.show()


# Verificando valores outliers que foram padronizados em um único valor máximo
# Nota-se que a quantidade desses outliers corresponde a 4% com 965 valores do total da base.
outliers = df[df['median_house_value'] == df['median_house_value'].max()] 
(len(outliers) / len(df)) * 100

# Para observar se o modelo performa bem sem esses valores, é possível aplicar técnicas para excluir esses outliers, 
# como pegar o percentil até o valor que antece esse outlier. Portanto, vamos aplicar essa técnica em uma cópia da base.
# Nesse trecho de código é coletado todas as colunas númericas para filtrar até o percentil 99, eliminando possíveis outliers que venham prejudicar a qualidade da base.
df_manipulated = df.copy()
numeric_columns = df.select_dtypes(include='number').columns
for col in numeric_columns:
    df_manipulated = df_manipulated[df_manipulated[col] < df[col].quantile(0.99)]


df_manipulated.info()
1 - (len(df_manipulated) / len(df))


df_manipulated.describe()
sns.pairplot(df_manipulated, diag_kind='kde', plot_kws=dict(alpha=0.2))
plt.show()

fig, axis = plt.subplots(4, 3)
for ax, col in zip(axis.flatten(), df_manipulated.select_dtypes('number').columns):
    sns.boxplot(data=df_manipulated, x=col, ax=ax, showmeans=True)

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(df_manipulated.select_dtypes('number').corr(), dtype=bool))
sns.heatmap(
    df_manipulated.select_dtypes('number').corr(),
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    ax=ax
)
plt.show(block=False)


df_manipulated['ocean_proximity'].value_counts()
df_manipulated.info()
df_manipulated = df_manipulated.loc[df_manipulated['ocean_proximity'] != "ISLAND"]
df_manipulated['ocean_proximity'] = df_manipulated['ocean_proximity'].astype('category')


int_columns = [col for col in df_manipulated.select_dtypes('number').columns 
               if df_manipulated[col].apply(float.is_integer).all()]
float_columns = df_manipulated.select_dtypes('number').columns.difference(int_columns)


df_manipulated.info()

df_manipulated[int_columns] = df_manipulated[int_columns].apply(
    pd.to_numeric, downcast='integer'
)

df_manipulated[float_columns] = df_manipulated[float_columns].apply(
    pd.to_numeric, downcast='float'
)

df_manipulated.info()
df_manipulated.describe()


# Para manter a otimização da tabela, o dataframe será exportado com o tipo parquet.
df_manipulated.to_parquet('housing_manipulated.parquet', index=False)



