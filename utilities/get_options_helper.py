import getopt, sys

class GetOptionsHelper:
    def __init__(self, arguments = []):
        try:
            self.operations, self.arguments = getopt.getopt(sys.argv[1:], "-", arguments)
        except getopt.GetoptError as exception:
            self.operations, self.arguments = [], []

    # Função utilizada para pegar um parâmetro passado
    def get(self, operator, default_value=None, empty_value=None):
        for operation in self.operations:
            if operation[0] == operator:
                return empty_value if len(operation[1]) == 0 else operation[1]
            
        return default_value