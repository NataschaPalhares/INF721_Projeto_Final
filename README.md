# Classificação de Espécies de Aves

Este repositório conta com códigos que desenvolvem um modelo de classificação de espécies de aves com base em imagens. A motivação para o uso de RNAs para essa tarefa é devido sua grande capacidade em lidar com dados complexos, como imagens, e aprender padrões relevantes para a classificação. O objetivo é ser capaz de identificar corretamente e com alta precisão as espécies de aves com base em uma imagem, podendo ser utilizada, por exemplo, para identificar espécies de aves ameaçadas.


## Exemplos de Imagens do Conjunto de Dados

![1](https://github.com/NataschaPalhares/INF721_Projeto_Final/assets/88913342/91c7b9d0-20df-4b0c-b9fc-ac3392ea385e)
![1](https://github.com/NataschaPalhares/INF721_Projeto_Final/assets/88913342/176fa069-803c-4b35-a10e-b1d5851ad801)
![4](https://github.com/NataschaPalhares/INF721_Projeto_Final/assets/88913342/a0dffa27-4694-4ef2-aed6-c0be25849f2b)


## Bibliotecas Necessárias

```
pip install torch
pip install pandas
pip install numpy
pip install seaborn
pip install matplotlib
pip install torchvision
pip install scikit-learn
```

## Conjunto de Dados

https://www.kaggle.com/datasets/gpiosenka/100-bird-species/data

O dataset possui 89885 imagens de 525 espécies de aves e um arquivo csv.

Treinamento ≈ 94%   

Validação ≈ 3%  (5 imagens para cada espécie)

Teste ≈ 3% (5 imagens para cada espécie)


## Códigos

### inf721_cleaning.py

Corrige alguns erros e ajeita o arquivo csv disponibilizado junto com o conjunto de dados.

A variável numSpecies é responsável por definir quantas espécies das 525 serão utilizadas (lembrando que a contagem começa do zero, logo, numSpecies = 11, significa que estão sendo utilizadas 12 espécies).

### inf721_dataset.py

Define o conjunto de dados estendendo a classe Dataset e DataLoader do Pytorch.

### inf721_model.py

Define o modelo neural estendeno a classe Module do Pytorch. Foi implementado uma arquitetura de uma CNN.

### inf721_train.py

Executa o procedimento de treinamento e validação. Salva os pesos do modelo em 'trained_model.pth'. Exibe um gráfico com as curvas da loss do treinamento e da validação.

### inf721_inference.py

Carrega o modelo salvo, exibe a loss do teste e uma matriz de confusão.


## Execução

Após fazer o download do conjunto de dados e dos códigos desse repositório, certifique-se de que eles estão no mesmo diretório.

Definir em inf721_cleaning.py quantas espécies deseja usar, modificando apenas o número da váriavel numSpecies.

Executar o código de treinamento e depois o de inferência.

```
python3 inf721_train.py
```
```
python3 inf721_inference.py
```


## Exemplo de Resultados

### Utilizando 12 Espécies


Average Test Loss = 0.15743689984083176

Accuracy = 95%

![95](https://github.com/NataschaPalhares/INF721_Projeto_Final/assets/88913342/15bf750b-7182-42a6-8f8f-0a1d0291f225)
![12](https://github.com/NataschaPalhares/INF721_Projeto_Final/assets/88913342/8cf0092b-90fa-44ca-a12e-fa2e7d98a8e1)


### Utilizando 20 Espécies

Average Test Loss = 0.3744361959397793

Accuracy = 89%

![89](https://github.com/NataschaPalhares/INF721_Projeto_Final/assets/88913342/d140f715-3d84-4b90-a5b2-613c876d0142)
![20](https://github.com/NataschaPalhares/INF721_Projeto_Final/assets/88913342/8d334f7c-2a4b-40dc-9a9c-726206a13041)


### Utilizando 40 Espécies

Average Test Loss = 0.6449991890362331

Accuracy = 84,5%

![85](https://github.com/NataschaPalhares/INF721_Projeto_Final/assets/88913342/5f745724-a6c5-4313-bfe8-ee06ed696355)
![Captura de tela 2023-12-04 094214](https://github.com/NataschaPalhares/INF721_Projeto_Final/assets/88913342/e8549496-9c8d-4af5-97f2-d92758d6c769)


