import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from langchain import PromptTemplate
from langchain import LLMChain
from langchain_together import Together
import re
import pdfplumber
# Set the API key with double quotes

os.environ['TOGETHER_API_KEY'] = "5653bbfbaf1f7c1438206f18e5dfc2f5992b8f0b6aa9796b0131ea454648ccde"

text = ""
max_pages = 16
with pdfplumber.open("Ad_Mod_Daily_Sales.pdf") as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            text += page.extract_text() + "\n"

def Bot(Questions):
    chat_template = """
    Based on the provided context: {text}
    Please answer the following question: {Questions}
    Only provide answers that are directly related to the context. If the question is unrelated, respond with "I don't know".
    """
    prompt = PromptTemplate(
        input_variables=['text', 'Questions'],
        template=chat_template
    )
    llama3 = Together(model="meta-llama/Llama-3-70b-chat-hf", max_tokens=250)
    Generated_chat = LLMChain(llm=llama3, prompt=prompt)

    try:
        response = Generated_chat.invoke({
            "text": text,
            "Questions": Questions
        })

        response_text = response['text']

        response_text = response_text.replace("assistant", "")

        # Post-processing to handle repeated words and ensure completeness
        words = response_text.split()
        seen = set()
        filtered_words = [word for word in words if word.lower() not in seen and not seen.add(word.lower())]
        response_text = ' '.join(filtered_words)
        response_text = response_text.strip()  # Ensuring no extra spaces at the ends
        if not response_text.endswith('.'):
            response_text += '.'

        return response_text
    except Exception as e:
        return f"Error in generating response: {e}"

def ChatBot(Questions):
  greetings = ["hi", "hello", "hey", "greetings", "what's up", "howdy"]
    # Check if the input question is a greeting
  question_lower = Questions.lower().strip()
  if question_lower in greetings or any(question_lower.startswith(greeting) for greeting in greetings):
        return "Hello! How can I assist you with the document today?"
  else:
    response=Bot(Questions)
    return response.translate(str.maketrans('', '', '\n'))
  # text_embedding = model.encode(text, convert_to_tensor=True)
  # statement_embedding = model.encode(statement, convert_to_tensor=True)

  # # Compute the cosine similarity between the embeddings
  # similarity = util.pytorch_cos_sim(text_embedding, statement_embedding)

  # # Print the similarity score
  # print(f"Cosine similarity: {similarity.item()}")

  # # Define a threshold for considering the statement as related
  # threshold = 0.7

  # if similarity.item() > threshold:
  #   response=Bot(Questions)
  #   return response
  # else:
  #   response="The statement is not related to the text."
  #   return response

iface = gr.Interface(fn=ChatBot, inputs="text", outputs="text", title="Chatbot")
iface.launch(debug=True)

