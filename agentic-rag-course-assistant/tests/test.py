from app.agent.llm import QwenLLM

llm = QwenLLM()

question = "Explique le mécanisme de self-attention."

print("Question :", question)
print("Réponse :", llm.generate(question))