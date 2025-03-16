import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Zamień "username/model-name" na nazwę Twojego modelu
model_name = "gorni123/orkibotllm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chatbot_response(user_input):
    # Tokenizuj wejście
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    # Generuj odpowiedź (dostosuj parametry, np. max_length)
    output_ids = model.generate(input_ids, max_length=150, do_sample=True, top_p=0.9)
    # Dekoduj wygenerowaną odpowiedź
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Definicja interfejsu za pomocą Gradio
iface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Wpisz wiadomość..."),
    outputs="text",
    title="Mój Chatbot",
    description="Chatbot zbudowany na bazie mojego modelu AI."
)

if __name__ == "__main__":
    iface.launch()
