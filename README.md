# IA e Análise de Dados com Xadrez - Projeto Universitário

## Visão Geral
Este é um projeto acadêmico desenvolvido em **Python** como parte do curso **Ciência da Computação na Universidade São Judas Tadeu**. O objetivo deste projeto é desenvolver um protótipo de IA, onde dado um determinado matchup, seria realizada uma análise pelo código para que então pudesse dizer qual dos jogadores tem a maior probabilidade de vitória. Esta análise é realizada utilizando-se de um histórico de partidas, onde diversas informações já presentes e não presentes (que são criadas quando necessário) são utilizadas para aprimorar e treinar os modelos de IA.

## Dependências
- joblib==1.3.2
- numpy==1.26.1
- pandas==2.1.1
- python-dateutil==2.8.2
- pytz==2023.3.post1
- scikit-learn==1.3.2
- scipy==1.11.3
- six==1.16.0
- threadpoolctl==3.2.0
- tzdata==2023.3

## Pré-requisitos
Antes de começar, certifique-se de ter instalado:
- Ter o [Python](https://www.python.org/downloads/) instalado na sua máquina

## Instalação
1. Clone o repositório: `git clone https://github.com/gmdvv2003/projeto-a322023-IA-xadrez.git` e depois navegue até o diretório do projeto
2. Instale as dependências: `pip install -r requirements.txt` ou utilize da lista de dependências acima

## Uso
Antes de prosseguir é importante que o arquivo [games.csv](https://www.kaggle.com/datasets/datasnaek/chess) esteja presente no diretório do projeto com o nome correto.

Depois de clonado, para executar o código basta:
1. Abrir o CMD e navegar até o projeto:
```
cd caminho/do/projeto
```
2. Digitar a seguinte linha de comando:
```
py main.py
```

Esse único comando já é o suficiente para executar o código.
Também há algumas `flags` na qual você pode fazer uso para facilitar/alterar o método de execução do código.

- `--mskip` Para pular o prompt inicial de modificação do arquivo `games.csv`
- `--w` e `--b` Para pular o prompt de inserção dos nomes dos jogadores (Caso os mesmos sejam válidos)
- `--rw` e `--rb` Para rodar os modelos em jogadores brancos e pretos aleatórios respectivamente
- `--gskip` Para pular a execução do GridSearchCV
- `--sskip` Para pular a criação do arquivo com as melhores amostras

Um exemplo de código que simplesmente rodaria os modelos sem etapas intermediárias seria:
```
py main.py --mskip --rw --rb --gskip --sskip
```

## Licença
Os direitos do projeto estão reservados perante a licença "CC BY" (Creative Commons Attribution), que permite o uso, redistribuição e modificação do trabalho, desde que seja atribuido crédito aos autores originais.

## Equipe

| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/99232385?v=4" width=115><br><sub>Stefani Marchi</sub>](https://github.com/stefanimarchi)  |  [<img loading="lazy" src="https://avatars.githubusercontent.com/u/140629456?v=4" width=115><br><sub>Guilherme Daghlian</sub>](https://github.com/gmdvv2003)  |  [<img loading="lazy" src="https://avatars.githubusercontent.com/u/72100162?v=4" width=115><br><sub>Victor Hugo</sub>](https://github.com/Victor733)  |  [<img loading="lazy" src="https://avatars.githubusercontent.com/u/121135887?v=4" width=115><br><sub>Gustavo Gino</sub>](https://github.com/GustavoGinoTerezo)  |  [<img loading="lazy" src="https://avatars.githubusercontent.com/u/135339704?v=4" width=115><br><sub>Diego Gallardo</sub>](https://github.com/Dieguinn19)
| :---: | :---: | :---: | :---: | :---: |
