import streamlit as st
import tensorflow as tf
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor, AutoTokenizer
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

tokenizer2 = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base",
                                             return_dict=True)
model_path = "./Models/T5_gen"
device = torch.device('cpu')

# Load the model and map it to the CPU using the specified device
Model1 = torch.load(model_path, map_location=device)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
generator = pipeline('text-generation',model='./Models/GPT-2_gen', tokenizer='gpt2')


def generate_commentary(text):
    model.eval()
    input_ids = tokenizer2.encode("WebNLG:{} </s>".format(text), return_tensors="pt")  # Batch size 1
    input_ids = input_ids.to(dev)
    s = time.time()
    outputs = Model1.generate(input_ids)
    gen_text=tokenizer2.decode(outputs[0]).replace('<pad>','').replace('</s>','')
    elapsed = time.time() - s
    print('Generated in {} seconds'.format(str(elapsed)[:4]))
    return gen_text
def listToString(ab):
    p=[]
    for key, val in ab[0].items():
        p.append("{}".format(val))
    str1 = ""
    for ele in p:
        str1 += ele
    res = ""
    cnt = 0
    for ch in str1:
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
