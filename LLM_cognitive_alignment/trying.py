from litgpt import LLM

llm = LLM.load("meta-llama/Meta-Llama-3.1-8B")
text = llm.generate("Question: Who was the first president of the United States? Choices: (A) Barack Obama (B) George Washington (C) Michael Jackson Answer:")
print(text)


