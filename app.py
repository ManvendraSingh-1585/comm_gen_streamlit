import streamlit as st
import tensorflow as tf
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor, AutoTokenizer,GPT2Tokenizer, GPT2LMHeadModel
from sklearn.utils import shuffle
import time
from transformers import TextDataset,DataCollatorForLanguageModeling,pipeline
from gtts import gTTS
from IPython.display import Audio

# Load your ML model and any other required components
if torch.cuda.is_available():
   dev = torch.device("cuda:0")
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

model_path_t5 = "https://drive.google.com/file/d/1--Ls6VzRSq7tOS5P2XSbvBFZxi8Zm9iL/view?usp=sharing"
tokenizer2 = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
model.load_state_dict(torch.load(model_path_t5, map_location=dev))
model.to(dev)

# Load your GPT-2 model and any other required components
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_path_gpt2 = "https://drive.google.com/file/d/1-fdo3tedGIaQRby8RDGQ9HfNJdUgreBT/view?usp=sharing"
model2 = GPT2LMHeadModel.from_pretrained("gpt2")
model2.load_state_dict(torch.load(model_path_gpt2, map_location=dev))
model2.to(dev)



def generate_commentary(text):
    model.eval()
    input_ids = tokenizer2.encode("WebNLG:{} </s>".format(text), return_tensors="pt")  # Batch size 1
    input_ids = input_ids.to(dev)
    s = time.time()
    outputs = model.generate(input_ids)
    gen_text=tokenizer2.decode(outputs[0]).replace('<pad>','').replace('</s>','')
    elapsed = time.time() - s
    print('Generated in {} seconds'.format(str(elapsed)[:4]))
    return gen_text
def generator(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)
def listToString(ab):
    res = ""
    cnt = 0
    for ch in ab:
        res += ch
        if ch == '.':
            cnt += 1
            if cnt == 4:
                break
    if res[-1]!='.':
        res+='.'
    return res

def synthesize_audio(commentary):
    gTTS_object = gTTS(text = commentary,lang = "en",slow = False)
    gTTS_object.save("C:\\Users\\manve\\Desktop\\Audio_Output_Commentary\\gtts.wav")

# Streamlit app
def main():
    st.title("Commentary Generator")
    input_text = st.text_input("Enter text")

    if st.button("Generate Commentary"):
        # Generate commentary based on input text
        generated_commentary = generator(generate_commentary(input_text))

        res = listToString(generated_commentary)

        # Display the generated commentary
        st.markdown("## Generated Commentary")
        st.write(res)

        # Synthesize audio from generated commentary
        synthesize_audio(res)

        # Save the audio file
        audio_file_path = "C:\\Users\\manve\\Desktop\\Audio_Output_Commentary\\gtts.wav"  
        # Set your desired path
        # Save audio file using appropriate library function
        
        # Provide option to play audio
        st.audio(audio_file_path)

if __name__ == "__main__":
    main()
