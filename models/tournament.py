from utilities.execute_game import execute_game

import math
import time

def main():
    # Quantidade de jogadores que irão participar do torneio
    TOTAL_PLAYERS = int(input("Digite a quantidade de jogadores que devem participar do torneio (Deve ser um número aplicável a base 2): "))
    if TOTAL_PLAYERS <= 1 or math.log2(TOTAL_PLAYERS)%1 != 0:
        print("A quantidade de jogadores deve ser maior que 1 e deve ser aplicável a base 2 (2, 4, 8, 16, 32, 64...)")
    else:
        # Benchmarking
        start_time = time.time()

        # Total de jogos
        total_games = TOTAL_PLAYERS - 1

        # Chaves dos jogos
        games_bracket = [[None, None, None] for _ in range(total_games)]

        for game in range(total_games):
            # Fase em que o jogo atual se encontra
            current_phase = math.ceil(math.log2(TOTAL_PLAYERS) - math.log(total_games - game, 2)) - 1

            white_player_id = None
            black_player_id = None

            if current_phase > 0:
                # Soma da quantidade de jogos até a fase anterior e a fase atual
                games_up_to_last_phase = TOTAL_PLAYERS - (TOTAL_PLAYERS/(2**(current_phase - 1)))
                games_up_to_current_phase = TOTAL_PLAYERS - (TOTAL_PLAYERS/(2**current_phase))

                # Posição dos vencedores da última fase de grupos
                group_position = int(games_up_to_last_phase + (game%games_up_to_current_phase)*2)

                # Atualiza os IDs para serem correspondentes aos vencedores das fases de grupos anteriores
                white_player_id = games_bracket[group_position][2]
                black_player_id = games_bracket[group_position + 1][2]

                print(f"\nExecutando o {game + 1}º jogo entre {white_player_id} e {black_player_id}")
            else:
                print(f"\nExecutando o {game + 1}º jogo.")

            # Executa uma partida com os parâmetros passados
            execution_success, execution_result = execute_game(["--mskip", "--" + (f"w={white_player_id}" if white_player_id else "rw"), "--" + (f"b={black_player_id}" if black_player_id else "rb"), "--gskip"])

            if not execution_success:
                pass
            else:
                # Variáveis para verificar qual foi o vencedor mais comum (caso haja diferença nos resultados dos modelos)
                total_white_wins = 0
                total_black_wins = 0

                for name, result in execution_result.items():
                    if result["winner"] == "white":
                        total_white_wins += 1
                        print(f"\tVencedor do modelo {name} foi a cor branca, jogador: {result['white_player_id']}")
                    elif result["winner"] == "black":
                        total_black_wins += 1
                        print(f"\tVencedor do modelo {name} foi a cor preta, jogador: {result['black_player_id']}")

                    white_player_id = result["white_player_id"]
                    black_player_id = result["black_player_id"]
                
                print(f"\nJogo entre {white_player_id} vs {black_player_id} finalizado.")
                print(f"Vencedor com a maior quantidade de vitórias foi: {white_player_id if total_white_wins > total_black_wins else black_player_id}")

                games_bracket[game][0] = white_player_id
                games_bracket[game][1] = black_player_id
                games_bracket[game][2] = white_player_id if total_white_wins > total_black_wins else black_player_id

                print(f"\n{'#'*100}")

        print(f"\nTorneio entre {TOTAL_PLAYERS} finalizado depois de {time.time() - start_time} segundos.")
        print(f"Vencedor do torneio foi o jogador: {games_bracket[total_games - 1][2]}")

if __name__ == "__main__":
    main()