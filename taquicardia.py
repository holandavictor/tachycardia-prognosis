# -*- coding: utf-8 -*-


from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


df = pd.read_csv("healthcare-dataset-stroke-data.csv", low_memory=False)
print(len(df.index), 'dataset original')

#remove rows with any nan value
df = df.dropna()

print(len(df.index), 'sem nan')
print(len(df.columns), 'com todas colunas')


# Exclusão da coluna de id dos pacientes, se foi casado, tipo de trabalho
df = df.drop(['id'], axis=1)
df = df.drop(['ever_married'], axis=1)
df = df.drop(['work_type'], axis=1)


print(len(df.columns), 'sem colunas removidas {id, married, work type}')

# Relação entre o tipo de residencia e o nível de estresse
df.rename(columns = {'Residence_type':'stress'}, inplace = True)
df = df.replace('Urban', 1)
df = df.replace('Rural', 0)

# remove rows with gender value 'Other'
df = df[df['gender'] != 'Other']


print(len(df.index), 'removendo genero other')


# Exclusão de instância que não tem a info de smoke
df = df[df['smoking_status'] != 'Unknown']


print(len(df.index), 'removendo pacientes sem informação sobre tabagismo')


### preprocess original data ####
original_df = df



# Gender
df = df.replace('Male', 0)
df = df.replace('Female', 1)

# Set score according age of pacient cha2ds2-vasc
df.loc[df.age < 65, 'age'] = 0
df.loc[((df.age >= 65) & (df.age < 75)), 'age'] = 1
df.loc[df.age >= 75, 'age'] = 2


#categoriza diabetes de acordo com o nível de glicose
df['avg_glucose_level'] = np.where((df.avg_glucose_level < 126),0,df.avg_glucose_level) 
df['avg_glucose_level'] = np.where((df.avg_glucose_level >= 126),1,df.avg_glucose_level) #Diabetes


# Set score according smoking of pacient
df['smoking_status'] = np.where((df.smoking_status == 'smokes'),2,df.smoking_status) #Fumante
df['smoking_status'] = np.where((df.smoking_status == 'formerly smoked'),1,df.smoking_status) #Ex Fumante
df['smoking_status'] = np.where((df.smoking_status == 'never smoked'),0,df.smoking_status)


# categorizar bmi para se o valor for entre 25 e 30 é sobrepeso
df['bmi'] = np.where((df.bmi < 25),0,df.bmi)
df['bmi'] = np.where(((df.bmi >= 25) & (df.bmi <= 30)),1,df.bmi) #Sobrepeso
df['bmi'] = np.where((df.bmi > 30),2,df.bmi) #Obesidade





# renomear nomes de colunas
df.rename(columns = {'bmi':'imc'}, inplace = True)
df.rename(columns = {'stroke':'avc'}, inplace = True)
df.rename(columns = {'age':'idade'}, inplace = True)
df.rename(columns = {'gender':'sexo'}, inplace = True)
df.rename(columns = {'hypertension':'hipertensao'}, inplace = True)
df.rename(columns = {'heart_disease':'doenca_vascular'}, inplace = True)
df.rename(columns = {'stress':'estresse'}, inplace = True)
df.rename(columns = {'smoking_status':'tabagismo'}, inplace = True)
df.rename(columns = {'avg_glucose_level':'diabetes'}, inplace = True)


# avc to 2
df['avc'] = np.where((df.avc == 1),2,df.avc)


#extrair variável insuficiencia cardíaca a partir das informações obtidas no dataset
#(Hipertensão, Idade >= 70, Tabagismo, Diabetes (genero(2x homem, 4x mulher)), Obesidade, AVC) = % ic (insuficiencia cardiaca)



# cria copia das colunas de interesse, com excecao da coluna age (vamos pegar do dataset original)
ic_calc = df[['sexo', 'idade', 'hipertensao', 'tabagismo', 'diabetes', 'imc', 'avc']].copy()



# processamento do score de diabetes baseado no sexo
ic_calc['novo_diabetes'] = 0
ic_calc['novo_diabetes'] = np.where(((ic_calc.sexo == 1) & (ic_calc['diabetes'] == 1)),2,ic_calc.novo_diabetes) # se mulher e diabetes = 2
ic_calc['novo_diabetes'] = np.where(((ic_calc.sexo == 0) & (ic_calc['diabetes'] == 1)),1,ic_calc.novo_diabetes) # se homem e diabetes = 1



# remove as colunas que nao vao ser mais usadas
ic_calc = ic_calc.drop(['sexo'], axis=1)
ic_calc = ic_calc.drop(['diabetes'], axis=1)



# renomear coluna diabetes
ic_calc.rename(columns = {'novo_diabetes':'diabetes'}, inplace = True)



# tratamento do score da idade baseado no limiar 70
# ic_calc.loc[ic_calc.idade < 70, 'idade'] = 0
# ic_calc.loc[ic_calc.idade >= 70, 'idade'] = 1



# refazer score do avc
# ic_calc.loc[ic_calc.avc == 2, 'avc'] = 1



# transformando ex fumante
# ic_calc.loc[ic_calc.tabagismo == 2, 'tabagismo'] = 1


#Soma variaveis para calcular ic
ic_calc['score'] = ic_calc['hipertensao'] + ic_calc['idade'] + ic_calc['avc'] + ic_calc['diabetes'] + ic_calc['imc'] + ic_calc['tabagismo']

column = 'score'
ic_calc[column] = (ic_calc[column] - ic_calc[column].min()) / (ic_calc[column].max()- ic_calc[column].min())
#verifica peso das variaveis
pca = PCA(n_components=1)
pca.fit(ic_calc)
pca_resultado = pca.components_

#Define corte para classificar IC
ic_index = 0.5
#remove aqueles com index inferior a 0.5
ic_calc = ic_calc.drop(ic_calc[ic_calc.score < ic_index].index)

new_df_index_array = ic_calc.index.tolist()
df = df.filter(items = new_df_index_array, axis=0)
#atribui novo campo de ic com valor 1, 
#sem ic ja foi removido
df['ic'] = 1



# converte columns to numeric
df["sexo"] = pd.to_numeric(df["sexo"])
df["tabagismo"] = pd.to_numeric(df["tabagismo"])



# reordenar colunas
# df = df[['ic', 'hipertensao', 'idade', 'diabetes', 'avc', 'doenca_vascular', 'sexo', 'estresse', 'imc', 'tabagismo']]




# create new column adapted chadvasc
df['cha2ds2_vasc'] = df['hipertensao'] + df['idade'] + df['avc'] + df['doenca_vascular'] + df['sexo'] + df['ic']
df['habitos'] = df['estresse'] + df['tabagismo'] + df['imc']


#novas colunas relacionados ao risco de acordo com chadvasc


#CLASSIFICAÇÃO CHA2DS2-VASc (Friberg 2012)
df['f_risco_avc_isquemico_anual'] = 0
df['f_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 0),0.2,df.f_risco_avc_isquemico_anual)
df['f_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 1),0.6,df.f_risco_avc_isquemico_anual)
df['f_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 2),2.2,df.f_risco_avc_isquemico_anual)
df['f_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 3),3.2,df.f_risco_avc_isquemico_anual)
df['f_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 4),4.8,df.f_risco_avc_isquemico_anual)
df['f_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 5),7.2,df.f_risco_avc_isquemico_anual)
df['f_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 6),9.7,df.f_risco_avc_isquemico_anual)
df['f_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 7),11.2,df.f_risco_avc_isquemico_anual)
df['f_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 8),10.8,df.f_risco_avc_isquemico_anual)
df['f_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 9),12.2,df.f_risco_avc_isquemico_anual)

df['f_risco_avc_ait_embolia_sistemica'] = 0
df['f_risco_avc_ait_embolia_sistemica'] = np.where((df.cha2ds2_vasc == 0),0.3,df.f_risco_avc_ait_embolia_sistemica)
df['f_risco_avc_ait_embolia_sistemica'] = np.where((df.cha2ds2_vasc == 1),0.9,df.f_risco_avc_ait_embolia_sistemica)
df['f_risco_avc_ait_embolia_sistemica'] = np.where((df.cha2ds2_vasc == 2),2.9,df.f_risco_avc_ait_embolia_sistemica)
df['f_risco_avc_ait_embolia_sistemica'] = np.where((df.cha2ds2_vasc == 3),4.8,df.f_risco_avc_ait_embolia_sistemica)
df['f_risco_avc_ait_embolia_sistemica'] = np.where((df.cha2ds2_vasc == 4),6.7,df.f_risco_avc_ait_embolia_sistemica)
df['f_risco_avc_ait_embolia_sistemica'] = np.where((df.cha2ds2_vasc == 5),10.0,df.f_risco_avc_ait_embolia_sistemica)
df['f_risco_avc_ait_embolia_sistemica'] = np.where((df.cha2ds2_vasc == 6),13.6,df.f_risco_avc_ait_embolia_sistemica)
df['f_risco_avc_ait_embolia_sistemica'] = np.where((df.cha2ds2_vasc == 7),15.2,df.f_risco_avc_ait_embolia_sistemica)
df['f_risco_avc_ait_embolia_sistemica'] = np.where((df.cha2ds2_vasc == 8),15.7,df.f_risco_avc_ait_embolia_sistemica)
df['f_risco_avc_ait_embolia_sistemica'] = np.where((df.cha2ds2_vasc == 9),17.4,df.f_risco_avc_ait_embolia_sistemica)



#CLASSIFICAÇÃO CHA2DS2-VASc (Lip 2010)
df['l_risco_avc_isquemico_anual'] = 0
df['l_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 0),0.0,df.l_risco_avc_isquemico_anual)
df['l_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 1),0.6,df.l_risco_avc_isquemico_anual)
df['l_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 2),1.6,df.l_risco_avc_isquemico_anual)
df['l_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 3),3.9,df.l_risco_avc_isquemico_anual)
df['l_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 4),1.9,df.l_risco_avc_isquemico_anual)
df['l_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 5),3.2,df.l_risco_avc_isquemico_anual)
df['l_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 6),3.6,df.l_risco_avc_isquemico_anual)
df['l_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 7),8.0,df.l_risco_avc_isquemico_anual)
df['l_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 8),11.1,df.l_risco_avc_isquemico_anual)
df['l_risco_avc_isquemico_anual'] = np.where((df.cha2ds2_vasc == 9),100,df.l_risco_avc_isquemico_anual)





df['ic'] = df['ic'].astype("int")
df['avc'] = df['avc'].astype("int")
df['imc'] = df['imc'].astype("int")
df['idade'] = df['idade'].astype("int")
df['sexo'] = df['sexo'].astype("int")
df['hipertensao'] = df['hipertensao'].astype("int")
df['doenca_vascular'] = df['doenca_vascular'].astype("int")
df['estresse'] = df['estresse'].astype("int")
df['tabagismo'] = df['tabagismo'].astype("int")
df['diabetes'] = df['diabetes'].astype("int")





# resultado da soma
df['resultado_score'] = (df['cha2ds2_vasc'] * 0.7) + (df['habitos'] * 0.3)
df['resultado_normalizado'] = df['resultado_score']


# # normalizar resultados <<<<
column = 'resultado_normalizado'
min_value = 0;
max_value = 13; #atribuir maior obtido pelos scores
df[column] = (df[column] - df[column].min()) / (df[column].max()- df[column].min())
df[column] = df[column] * 10


pca.fit(df)
pca_resultado_df = pca.components_

# cria novas categorias para inferencia
df.loc[df['resultado_normalizado'] <= 2.5, 'resultado_categoria'] = 'I'
df.loc[(df['resultado_normalizado'] > 2.5) & (df['resultado_normalizado'] <= 5), 'resultado_categoria'] = 'II'
df.loc[(df['resultado_normalizado'] > 5) & (df['resultado_normalizado'] <= 7.5), 'resultado_categoria'] = 'III'
df.loc[df['resultado_normalizado'] > 7.5, 'resultado_categoria'] = 'IV'

# Atribui mortalidade com base na categoria (Sem tratamento)
df['mort_sem_tra_min']= 0
df['mort_sem_tra_max'] = 0
df['mort_sem_tra_min'] = np.where((df.resultado_categoria == 'I'),0.05,df.mort_sem_tra_min)
df['mort_sem_tra_max'] = np.where((df.resultado_categoria == 'I'),0.19,df.mort_sem_tra_max)

df['mort_sem_tra_min'] = np.where((df.resultado_categoria == 'II'),0.15,df.mort_sem_tra_min)
df['mort_sem_tra_max'] = np.where((df.resultado_categoria == 'II'),0.40,df.mort_sem_tra_max)

df['mort_sem_tra_min'] = np.where((df.resultado_categoria == 'III'),0.15,df.mort_sem_tra_min)
df['mort_sem_tra_max'] = np.where((df.resultado_categoria == 'III'),0.40,df.mort_sem_tra_max)

df['mort_sem_tra_min'] = np.where((df.resultado_categoria == 'IV'),0.44,df.mort_sem_tra_min)
df['mort_sem_tra_max'] = np.where((df.resultado_categoria == 'IV'),0.66,df.mort_sem_tra_max)


# Atribui mortalidade com base na categoria (Com tratamento)
df['mort_com_tra_min'] = 0
df['mort_com_tra_max'] = 0
df['mort_com_tra_min'] = np.where((df.resultado_categoria == 'I'),0.05,df.mort_com_tra_min)
df['mort_com_tra_max'] = np.where((df.resultado_categoria == 'I'),0.10,df.mort_com_tra_max)

df['mort_com_tra_min'] = np.where((df.resultado_categoria == 'II'),0.05,df.mort_com_tra_min)
df['mort_com_tra_max'] = np.where((df.resultado_categoria == 'II'),0.10,df.mort_com_tra_max)

df['mort_com_tra_min'] = np.where((df.resultado_categoria == 'III'),0.10,df.mort_com_tra_min)
df['mort_com_tra_max'] = np.where((df.resultado_categoria == 'III'),0.15,df.mort_com_tra_max)

df['mort_com_tra_min'] = np.where((df.resultado_categoria == 'IV'),0.30,df.mort_com_tra_min)
df['mort_com_tra_max'] = np.where((df.resultado_categoria == 'IV'),0.40,df.mort_com_tra_max)


# Atribui mortalidade com base na categoria (Science Direct)
df['nyha_mortality_min'] = 0
df['nyha_mortality_max'] = 0
df['nyha_mort_com_tra_min'] = np.where((df.resultado_categoria == 'I'),0.10,df.mort_com_tra_min)
df['nyha_mortality_max'] = np.where((df.resultado_categoria == 'I'),0.15,df.mort_com_tra_max)

df['nyha_mortality_min'] = np.where((df.resultado_categoria == 'II'),0.10,df.mort_com_tra_min)
df['nyha_mortality_max'] = np.where((df.resultado_categoria == 'II'),0.15,df.mort_com_tra_max)

df['nyha_mortality_min'] = np.where((df.resultado_categoria == 'III'),0.15,df.mort_com_tra_min)
df['nyha_mortality_max'] = np.where((df.resultado_categoria == 'III'),0.20,df.mort_com_tra_max)

df['nyha_mortality_min'] = np.where((df.resultado_categoria == 'IV'),0.20,df.mort_com_tra_min)
df['nyha_mortality_max'] = np.where((df.resultado_categoria == 'IV'),0.50,df.mort_com_tra_max)



print('categoria I -> ', df[df['resultado_categoria'] == 'I'].shape[0])
print('categoria II -> ', df[df['resultado_categoria'] == 'II'].shape[0])
print('categoria III -> ', df[df['resultado_categoria'] == 'III'].shape[0])
print('categoria IV -> ', df[df['resultado_categoria'] == 'IV'].shape[0])

print(list(df.columns))

df.loc[:, ['ic', 'hipertensao', 'idade', 'diabetes', 'avc', 'doenca_vascular', 'sexo', 'cha2ds2_vasc', 
           'f_risco_avc_isquemico_anual', 'f_risco_avc_ait_embolia_sistemica', 'l_risco_avc_isquemico_anual',
           
           'estresse', 'imc', 'tabagismo', 'habitos', 
           'resultado_score', 'resultado_normalizado', 'resultado_categoria',
          
           'nyha_mortality_min','nyha_mortality_max',
           'mort_sem_tra_min', 'mort_sem_tra_max', 
           'mort_com_tra_min', 'mort_com_tra_max']]




df_saidas = df[['cha2ds2_vasc', 'habitos', 'resultado_normalizado','resultado_categoria','f_risco_avc_isquemico_anual','f_risco_avc_ait_embolia_sistemica', 'l_risco_avc_isquemico_anual', 'mort_sem_tra_min', 'mort_sem_tra_max', 'mort_com_tra_min', 'mort_com_tra_max']]
df_saidas.insert(0, 'n', df_saidas.index.tolist())
df_saidas.reset_index(drop=True, inplace = True)



# exportar dataframes
df.to_csv('preprocess_result.csv', sep=',', encoding='utf-8', index=False)
original_df.to_csv('original_df.csv', sep=',', encoding='utf-8', index=False)
ic_calc.to_csv('ic_calc.csv', sep=',', encoding='utf-8', index=False)
