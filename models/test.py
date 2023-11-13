from utilities.execute_game import execute_game
from utilities.get_options_helper import GetOptionsHelper

import time

# Total de testers a serem executados
TOTAL_TESTS = int(GetOptionsHelper(["t="]).get("--t", default_value=1))

def main() -> None:
    # Benchmarking dos tests em geral
    tests_start_time = time.time()

    # Dicionário com os resultados obtidos
    obtained_results = {}

    for game in range(TOTAL_TESTS):
        print(f"Executando teste de número: {game + 1}")

        # Benchmarking do tempo desta partida
        this_game_start_time = time.time()

        # Executa uma partida teste
        execution_success, execution_result = execute_game(["--mskip", "--rw", "--rb", "--gskip"])

        if not execution_success:
            pass
        else:
            # Adiciona os resultados desta partida a obtained_results
            for name, result in execution_result.items():
                if not (name in obtained_results):
                    obtained_results[name] = {
                        # Total de acurácia dos testes
                        "total_accuracy_score_value": 0,
                        "total_f1_score_value": 0,
                        "total_precision_score_value": 0,

                        # Total de vitórias de ambas as cores e/ou empates
                        "total_white_wins": 0,
                        "total_black_wins": 0,
                        "total_draws": 0
                    }

                # Dicionário com os resultados deste modelo
                this_model_results = obtained_results[name]

                # Adiciona as acúracias resultantes de cada estatística
                this_model_results["total_accuracy_score_value"] += result["accuracy_score_value"]
                this_model_results["total_f1_score_value"] += result["f1_score_value"]
                this_model_results["total_precision_score_value"] += result["precision_score_value"]

                # Adiciona o total de vitórias/empate de cada cor
                this_model_results[f"total_{'draws' if result['winner'] == 'draw' else 'white_wins' if result['winner'] == 'white' else 'black_wins'}"] += 1
        
        print(f"\tTeste finalizado depois de {time.time() - this_game_start_time} segundos.\n")

    print(f"{TOTAL_TESTS} teste(s) em {len(obtained_results)} modelo(s) diferente(s) finalizado(s) depois de {time.time() - tests_start_time} segundos.")

    # Printa os resultados finais dos modelos
    for name, result in obtained_results.items():
        print(f"\nMédia da acurácia do modelo {name}:")
        print(f"\t\"accuracy_score\": {result['total_accuracy_score_value']/TOTAL_TESTS*100}%")
        print(f"\t\"f1_score\": {result['total_f1_score_value']/TOTAL_TESTS*100}")
        print(f"\t\"total_precision_score_value\": {result['total_precision_score_value']/TOTAL_TESTS*100}\n")

        print(f"Vitórias e empates do modelo {name}:")
        print(f"\tBranco: {result['total_white_wins']}, Preto: {result['total_black_wins']}, Empates: {result['total_draws']}")

        print(f"\n{'#'*100}")

if __name__ == "__main__":
    main()