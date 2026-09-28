from typing import Protocol


class AnswerGenerator(Protocol):
    def generate(self, question: str, context: str) -> str: ...
