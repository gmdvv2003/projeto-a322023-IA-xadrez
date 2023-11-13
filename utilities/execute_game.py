import json
import os
import re
import subprocess

# Função utilizada para rodar um jogo em main.py e pegar o valor retornado
def execute_game(parameters: list[str] = []) -> [bool, dict | None]:
    # Caminho absoluto até o caminho do arquivo main.py
    working_dictory = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

    # Para garantir a existência da variável
    execution_process = None
    execution_result = None

    try:
        # Executa main.py utilizando um sub-processo
        execution_process = subprocess.run(["python", "main.py"] + parameters + ["--hp"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, cwd=working_dictory)
    except Exception as exception:
        print(f"Erro ao executar o sub-processo. Erro: {exception}")

    if execution_process != None:
        try:
            # Tenta transformar o resultado de volta à sua forma original
            execution_result = json.loads(re.findall(r"===HEADER===(.*?)===HEADER===", execution_process.stdout, re.DOTALL)[0].replace("\'", "\""))
            if not isinstance(execution_result, dict):
                raise TypeError("Resultado da execução não retornou um dicionário.")
        except Exception as exception:
            print(f"Erro ao acessar dados da execução. Erro: {exception}")

        if execution_result != None:
            return [True, execution_result]

    return [False, None]