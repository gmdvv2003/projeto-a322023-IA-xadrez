from utilities.get_options_helper import GetOptionsHelper

import numpy
import pandas
import time
import os

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Quantidade de amostras para utilizar no teste de hipótese
HYPOTHESIS_TEST_SAMPLE_SIZE = 30

# Porcentagem na qual os dados devem ser divididos em treino e teste
TEST_SPLIT_SIZE = 0.2

# Semente utilizada ao realizar testes (um número aleatório da mesma semente sempre ira produzir o mesmo valor)
RANDOM_STATE_SEED = 0

# Parâmetros iniciais dos modelos (valores gerados pelo GridSearchCV)
INITIAL_MODELS_PARAMETERS = {
    "HistGradientBoostingClassifier": {
        "random_state": RANDOM_STATE_SEED, "learning_rate": 0.01, "max_bins": 200, "max_depth": 5, "max_iter": 300, "min_samples_leaf": 2
    },
    "RandomForestClassifier": {
        "random_state": RANDOM_STATE_SEED, "min_samples_split": 10
    },
    "LogisticRegression": {
        "random_state": RANDOM_STATE_SEED, "max_iter": 150, "multi_class": "multinomial", "solver": "newton-cg"
    },
    "DecisionTreeClassifier": {
        "random_state": RANDOM_STATE_SEED, "max_features": "sqrt", "min_samples_leaf": 2, "min_samples_split": 5
    }
}

# Modelos na qual serão utilizados
candidate_models = {
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier(**INITIAL_MODELS_PARAMETERS["HistGradientBoostingClassifier"]),
    "RandomForestClassifier": RandomForestClassifier(**INITIAL_MODELS_PARAMETERS["RandomForestClassifier"]),
    "LogisticRegression": LogisticRegression(**INITIAL_MODELS_PARAMETERS["LogisticRegression"]),
    "DecisionTreeClassifier": DecisionTreeClassifier(**INITIAL_MODELS_PARAMETERS["DecisionTreeClassifier"])
}

# Range de valores que serão utilizados na hora de procurar os melhores parâmetros para o modelo
# Dependendo dos parâmetros o processo de análise pode demorar horas
candidate_models_grid_search_cv_parameters_range = {
    "HistGradientBoostingClassifier": {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_bins": [100, 200, 256],
        "max_depth": [3, 4, 5],
        "max_iter": [100, 200, 300],
        "min_samples_leaf": [1, 2, 4]
    },
    "RandomForestClassifier": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"]
    },
    "LogisticRegression": {
        "multi_class": ["multinomial"],
        "C": [0.01, 0.1, 1.0],
        "max_iter": [150, 300, 500],
        "penalty": ["l1", "l2", None],
        "solver": ["lbfgs", "saga", "newton-cg"]
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 10, 20],
        "max_features": ["auto", "sqrt", "log2"],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10]
    }
}

# Caso queira rodar o GridSearchCV pulando algum modelo, basta coloca-lo na lista abaixo
disabled_grid_search_cv_for_models = [
    "HistGradientBoostingClassifier",
    "RandomForestClassifier",
    "LogisticRegression",
    "DecisionTreeClassifier"
    ]

def main(skip_games_table_modification_prompt=False, white_match_up_id=None, black_match_up_id=None, skip_grid_search_cv=False, random_white_match_up_id=False, random_black_match_up_id=False, skip_best_samples_creation=False):
    # Garante a existência do arquivo games-modified.csv caso o mesmo não exista
    try:
        with open("./games-modified.csv", mode="x") as file:
            # Adiciona uma coluna template para que read_csv não falhe ao tentar ler o arquivo
            file.write("placeholder")
            file.close()
    except Exception as exception:
        pass

    # Lê o arquivo que contêm os jogos modificados
    games_table = pandas.read_csv("games-modified.csv")

    # Função utilizada para gerar um perfil do jogador, contendo algumas estatísticas
    def generate_player_profile_until(player_id, games_count):
        # Pega os primeiros n jogos
        player_games_table = games_table.head(1 + games_count)

        # Filtra a tabela para conter somente os jogos que player_id está incluso
        player_games_table = player_games_table[(player_games_table["white_id"] == player_id) | (player_games_table["black_id"] == player_id)]

        # Total de jogos
        total_games_played = len(player_games_table)

        # Total de vitorias e empates do jogador
        total_wins = len(player_games_table[(player_games_table["winner"] == "white") & (player_games_table["white_id"] == player_id) | (player_games_table["winner"] == "black") & (player_games_table["black_id"] == player_id)])
        total_draws = len(player_games_table[player_games_table["winner"] == "draw"])

        # Taxa de vitoria do jogador
        win_rate = (total_wins + 0.5*total_draws)/total_games_played if total_games_played > 0 else 1

        # Calcula a média do rating do jogador
        average_rating = sum([row["white_rating"] if row["white_id"] == player_id else row["black_rating"] for _, row in player_games_table.iterrows()])/total_games_played if total_games_played > 0 else 1

        return {
            "total_wins": [total_wins],
            "total_draws": [total_draws],
            "total_games_played": [total_games_played],
            "win_rate": [win_rate],
            "average_rating": [average_rating],
        }

    if not skip_games_table_modification_prompt and input("Você deseja atualizar e formatar os dados do arquivo? (S/N) ").lower() == "s":
        # Lê o arquivo dos jogos
        games_table = pandas.read_csv("games.csv")

        # Retira as colunas que não tem relevância
        games_table = games_table.drop(columns=[
            "id",
            "rated",
            "victory_status",
            "increment_code",
            "moves",
            "opening_eco",
            "opening_name",
            "opening_ply"
        ])

        # Todas os dados das tabelas abaixo são gerados POR partida 
        games_table = games_table.assign(white_total_games_played=[0]*len(games_table)) # Total de jogos pelo jogador branco
        games_table = games_table.assign(white_win_rate=[0]*len(games_table)) # Taxa de vitória do jogador branco
        games_table = games_table.assign(white_average_rating=[0]*len(games_table)) # Média do rating do jogador branco

        games_table = games_table.assign(black_total_games_played=[0]*len(games_table)) # Total de jogos pelo jogador preto
        games_table = games_table.assign(black_win_rate=[0]*len(games_table)) # Taxa de vitória do jogador preto
        games_table = games_table.assign(black_average_rating=[0]*len(games_table)) # Média do rating do jogador preto

        for row_index in range(0, len(games_table)):
            # Benchmarking
            start_time = time.time()

            # Linha no índice row_index contendo os dados da partida
            game_data = games_table.loc[row_index]

            # Perfil dos jogadores
            white_profile = generate_player_profile_until(game_data["white_id"], row_index)
            black_profile = generate_player_profile_until(game_data["black_id"], row_index)

            white_profile = {f"white_{key}": value for key, value in white_profile.items()}
            black_profile = {f"black_{key}": value for key, value in black_profile.items()}

            # Atualização da linha com os dados do perfil dos jogadores
            for key, value in white_profile.items():
                games_table.at[row_index, key] = value

            for key, value in black_profile.items():
                games_table.at[row_index, key] = value

            print(f"Linha: {row_index} modificada em {time.time() - start_time} segundos.")
                
        # Valor utilizado para determinar se a tabela foi modificada ou não
        games_table = games_table.assign(__modified__=[True]*len(games_table))

        # Atualiza o arquivo com os dados finais
        games_table.to_csv("games-modified.csv", index=False)
    else:
        # Verifica se o arquivo está no formato esperado ou não
        try:
            _ = games_table["__modified__"]
        except Exception:
            return [False, "Você deve atualizar e formatar o arquivo para para prosseguir."]

    if not skip_best_samples_creation:
        print("Gerando lista das melhores amostras...")

        # Lista somente os últimos jogos de cada jogador
        best_samples = pandas.DataFrame(columns=["id", "rating", "win_rate", "average_rating", "total_games_played", "total_wins", "total_draws"])
        
        for index, row in games_table.iterrows():
            white_id = row["white_id"]
            black_id = row["black_id"]

            white_profile = generate_player_profile_until(white_id, len(games_table))
            black_profile = generate_player_profile_until(black_id, len(games_table))
            
            # Atualiza os dados do jogador branco
            if (best_samples["id"] == white_id).any():
                best_samples[best_samples["id"] == white_id] = [white_id, row["white_rating"], row["white_win_rate"], row["white_average_rating"], row["white_total_games_played"], white_profile["total_wins"][0], white_profile["total_draws"][0]]
            else:
                best_samples.loc[index] = [white_id, row["white_rating"], row["white_win_rate"], row["white_average_rating"], row["white_total_games_played"], white_profile["total_wins"][0], white_profile["total_draws"][0]]

            # Atualiza os dados do jogador preto
            if (best_samples["id"] == black_id).any():
                best_samples[best_samples["id"] == black_id] = [black_id, row["black_rating"], row["black_win_rate"], row["black_average_rating"], row["black_total_games_played"], black_profile["total_wins"][0], black_profile["total_draws"][0]]
            else:
                best_samples.loc[index] = [black_id, row["black_rating"], row["black_win_rate"], row["black_average_rating"], row["black_total_games_played"], black_profile["total_wins"][0], black_profile["total_draws"][0]]

        print("Filtrando lista das melhores amostras...")

        # Filtra a lista para conter somente jogadores com mais de 10 jogos
        for index, row in best_samples.iterrows():
            if row["total_games_played"] < 10:
                best_samples = best_samples.drop(index=index)

        # Ordena a lista dos melhores amostras pelo rating
        best_samples = best_samples.sort_values(by=["total_games_played"], ascending=False)

        # Salva a lista das melhores amostras em um arquivo CSV
        best_samples.to_csv("best-samples.csv", index=False)

    best_samples = pandas.read_csv("best-samples.csv")

    top_n_players = best_samples.head(HYPOTHESIS_TEST_SAMPLE_SIZE)
    random_n_players = best_samples[~best_samples["id"].isin(top_n_players["id"])].sample(HYPOTHESIS_TEST_SAMPLE_SIZE)

    aSampleAverage = 0
    bSampleAverage = 0

    for _, row in top_n_players.iterrows():
        aSampleAverage += row["total_wins"]

    for _, row in random_n_players.iterrows():
        bSampleAverage += row["total_wins"]

    aSampleAverage /= HYPOTHESIS_TEST_SAMPLE_SIZE
    bSampleAverage /= HYPOTHESIS_TEST_SAMPLE_SIZE

    aNormalDistribution = 0
    bNormalDistribution = 0

    for _, row in top_n_players.iterrows():
        aNormalDistribution += (row["total_wins"] - aSampleAverage) ** 2

    for _, row in random_n_players.iterrows():
        bNormalDistribution += (row["total_wins"] - bSampleAverage) ** 2

    aNormalDistribution = (aNormalDistribution / (HYPOTHESIS_TEST_SAMPLE_SIZE - 1)) ** 0.5
    bNormalDistribution = (bNormalDistribution / (HYPOTHESIS_TEST_SAMPLE_SIZE - 1)) ** 0.5
    
    testValue = (aSampleAverage - bSampleAverage) / (((aNormalDistribution ** 2) / HYPOTHESIS_TEST_SAMPLE_SIZE + (bNormalDistribution ** 2) / HYPOTHESIS_TEST_SAMPLE_SIZE)) ** 0.5

    print(f"\nMédia de vitórias dos top {HYPOTHESIS_TEST_SAMPLE_SIZE} jogadores: {aSampleAverage}")
    print(f"Média de vitórias dos {HYPOTHESIS_TEST_SAMPLE_SIZE} jogadores aleatórios: {bSampleAverage}")

    print(f"Desvio padrão dos top {HYPOTHESIS_TEST_SAMPLE_SIZE} jogadores: {aNormalDistribution}")
    print(f"Desvio padrão dos {HYPOTHESIS_TEST_SAMPLE_SIZE} jogadores aleatórios: {bNormalDistribution}")

    print(f"Valor do teste: {testValue}\n")

    if random_white_match_up_id:
        white_match_up_id = games_table.sample()["white_id"].iloc[0]
        print(f"Jogador branco aleatório selecionado: {white_match_up_id}")

    if random_black_match_up_id:
        black_match_up_id = games_table.sample()["black_id"].iloc[0]
        print(f"Jogador preto aleatório selecionado: {black_match_up_id}")

    # Caso nenhum jogador tenha sido digitado, pede para o usuário digita-lo
    white_match_up_id = white_match_up_id or input("Digite o nome do primeiro jogador:\t")
    black_match_up_id = black_match_up_id or input("Digite o nome do segundo jogador:\t")

    # Verifica a existência dos IDs dos jogadores
    if not ((games_table["white_id"] == white_match_up_id) | (games_table["black_id"] == white_match_up_id)).any():
        return [False, f"\"{white_match_up_id}\" não é um ID de jogador valído."]
    
    if not ((games_table["white_id"] == black_match_up_id) | (games_table["black_id"] == black_match_up_id)).any():
        return [False, f"\"{black_match_up_id}\" não é um ID de jogador valído."]

    white_profile = generate_player_profile_until(white_match_up_id, len(games_table))
    black_profile = generate_player_profile_until(black_match_up_id, len(games_table))

    # Cria novas tabelas para guardar os IDs dos jogadores em forma de inteiros únicos
    ids_label = LabelEncoder()

    # Adiciona todos os IDs únicos de ambas as cores ao "Label" e converte os IDs
    ids_label.fit(numpy.concatenate([games_table["white_id"].to_list(), games_table["black_id"].to_list()]))

    games_table["white_id_label"] = ids_label.transform(games_table["white_id"])
    games_table["black_id_label"] = ids_label.transform(games_table["black_id"])

    # Pega as tabelas que contêm dados relevantes
    features = ["white_rating", "black_rating", "white_total_games_played", "white_win_rate", "white_average_rating", "black_total_games_played", "black_win_rate", "black_average_rating", "white_id_label", "black_id_label"]
    target = ["winner"]

    # Divide os dados em treinamento e teste
    x_train, x_test, y_train, y_test = train_test_split(games_table[features], games_table[target], test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE_SEED)

    # Tabela que contêm o jogo cujo resultado deve ser calculado
    game_to_predict = pandas.DataFrame({
        "white_rating": white_profile["average_rating"],
        "black_rating": black_profile["average_rating"],

        # Perfil do jogador branco
        "white_total_games_played": white_profile["total_games_played"],
        "white_win_rate": white_profile["win_rate"],
        "white_average_rating": white_profile["average_rating"],

        # Perfil do jogador preto
        "black_total_games_played": black_profile["total_games_played"],
        "black_win_rate": black_profile["win_rate"],
        "black_average_rating": black_profile["average_rating"],

        # IDs em formato de inteiro dos jogados branco e preto
        "white_id_label": [ids_label.transform([white_match_up_id])[0]],
        "black_id_label": [ids_label.transform([black_match_up_id])[0]],
    })

    # Resultado dos modelos do jogo
    models_results = {}

    for name, model in candidate_models.items():
        if not skip_grid_search_cv and not name in disabled_grid_search_cv_for_models:
            print(f"\nExecutando GridSearchCV no modelo {name}")

            # Benchmaking do GridSearchCV
            grid_search_cv_start_time = time.time()

            # Cria um GridSearchCV para o modelo
            grid_search = GridSearchCV(estimator=model, param_grid=candidate_models_grid_search_cv_parameters_range[name], scoring="accuracy", cv=5, n_jobs=-1)

            # Roda o GridSearchCV do modelo para encontrar os melhores parâmetros
            grid_search.fit(x_train, y_train.values.ravel())

            best_parameters = grid_search.best_params_
            best_model = grid_search.best_estimator_

            print(f"GridSearchCV do modelo {name} finalizado depois de {time.time() - grid_search_cv_start_time} segundos.")
            print(f"\tbest_parameters: {best_parameters}")
            print(f"\tbest_model: {best_model}")

            # Modifica os parâmetros do modelo para os melhores parâmetros encontrados
            model.set_params(**best_parameters)

        print(f"\nExecutando modelo {name}")

        # Benchmarking do treino
        training_start_time = time.time()

        # Treina o modelo utilizando os dados de treinamento
        model.fit(x_train, y_train.values.ravel())

        # Teste e verificação da acurácia do modelo
        test_prediction = model.predict(x_test)

        # Classificação das principais métricas
        classification_report_sheet = classification_report(test_prediction, y_test)

        print(f"Modelo {name} demorou {time.time() - training_start_time} segundos para ser executado.")
        print(f"\nPrincipais métricas utilizadas no modelo {name}:")
        print(classification_report_sheet)

        # Matrix de confusão do modelo
        confusion_matrix_sheet = confusion_matrix(test_prediction, y_test, labels=["black", "draw", "white"])

        print(f"Matrix de confusão do modelo {name}:")
        print("{:>16} {:>12} {:>12}\n".format("black", "draw", "white"))
        print("{:>5} {:>10} {:>12} {:>12}".format("black", *confusion_matrix_sheet[0]))
        print("{:>5} {:>10} {:>12} {:>12}".format("draw", *confusion_matrix_sheet[1]))
        print("{:>5} {:>10} {:>12} {:>12}".format("white", *confusion_matrix_sheet[2]))

        # Classificação da precisão
        accuracy_score_value = accuracy_score(y_test, model.predict(x_test))
        f1_score_value = f1_score(test_prediction, y_test, average="weighted")
        precision_score_value = precision_score(test_prediction, y_test, average="weighted")
        
        print(f"\nAcurácia do modelo {name}:")
        print(f"\t\"accuracy_score\": {accuracy_score_value*100:.2f}%")
        print(f"\t\"f1_score\": {f1_score_value*100:.2f}%")
        print(f"\t\"precision_score\": {precision_score_value*100:.2f}%\n")

        # ...e finalmente prevemos qual jogador tem a maior probabilidade de ganhar
        winner = model.predict(game_to_predict)[0]
        black_probability, draw_probability, white_probability = model.predict_proba(game_to_predict)[0]
        
        print(f"Probabilidades de vitoria pelo modelo {name}:\n\tPreta: {black_probability*100:.2f}%\n\tBranca: {white_probability*100:.2f}%\n\tEmpate: {draw_probability*100:.2f}%")

        # Adiciona o resultados do modelo 
        models_results[name] = {
            "white_player_id": white_match_up_id,
            "black_player_id": black_match_up_id,            

            # Acurácio do model
            "accuracy_score_value": accuracy_score_value,
            "f1_score_value": f1_score_value,
            "precision_score_value": precision_score_value,

            # Ganhador
            "winner": winner,

            # Probabilidades de vitória/empate
            "black_probability": black_probability,
            "draw_probability": draw_probability,
            "white_probability": white_probability
        }

        print(f"\n{'#'*100}")

    return [True, models_results]

if __name__ == "__main__":
    # Garante que o arquivo games.csv existe no diretório do projeto
    if not os.path.exists("./games.csv"):
        print("O arquivo \"games.csv\" não foi encontrado no diretório do projeto.")
    else:
        # Utilizado para facilitar a obtenção dos parâmetros passados
        options_getter = GetOptionsHelper(["mskip", "w=", "b=", "rw", "rb", "gskip", "sskip", "hp"])

        # Argumento para pular o prompt de modificação do arquivo games-modified.csv
        skip_games_table_modification_prompt = options_getter.get("--mskip", empty_value=True)

        # Argumento para inicializar com os jogadores selecionar
        white_match_up_id = options_getter.get("--w")
        black_match_up_id = options_getter.get("--b")

        # Argumento para inicializar com um jogador aleatório
        random_white_match_up_id = options_getter.get("--rw", empty_value=True)
        random_black_match_up_id = options_getter.get("--rb", empty_value=True)

        # Argumento para pular a execução do GridSearchCV nos modelos
        skip_grid_search_cv = options_getter.get("--gskip", empty_value=True)

        # Argumento para pular a criação do arquivo das melhores amostras
        skip_best_samples_creation = options_getter.get("--sskip", empty_value=True)

        # Executa o código com os parâmetros digitados
        success, execution_result = main(
            skip_games_table_modification_prompt=skip_games_table_modification_prompt,
            white_match_up_id=white_match_up_id,
            black_match_up_id=black_match_up_id,
            random_white_match_up_id=random_white_match_up_id,
            random_black_match_up_id=random_black_match_up_id,
            skip_grid_search_cv=skip_grid_search_cv,
            skip_best_samples_creation=skip_best_samples_creation
            )
        
        if not success:
            print(f"Algo de errado ocorreu enquanto o código era executado: {execution_result}")

        if options_getter.get("--hp", empty_value=True):
            print(f"===HEADER==={execution_result}===HEADER===")