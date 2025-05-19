import csv

class DataFetcher:
    def __init__(self):
        self.data = []

    def read_file(self, file_path: str):
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)

class QuestionFetcher(DataFetcher):
    def __init__(self) -> None:
        super().__init__()
        self.question: list[tuple[str, dict[int, dict[str, float]], str]] = []

    def parse(self, file_path: str):
        self.read_file(file_path)
        for row in self.data:
            trait_type = row['type']
            question = row['question']
            scores = {}
            for level in range(1, 6):
                key = f'score{level}'
                trait1_score, trait2_score = map(float, row[key].split('|'))
                scores[level] = {trait_type[0]: trait1_score, trait_type[1]: trait2_score}
            self.question.append((question, scores, trait_type))
