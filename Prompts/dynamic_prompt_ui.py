
import streamlit as st
from langchain_core.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
st.header('Research Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

template = load_prompt("dynamic_prompts/first_dynamic_prompt.json")
if st.button('Summarize'):
    prompt=template.invoke({"paper_input": paper_input, "style_input": style_input, "length_input": length_input})
    result=llm.invoke(prompt)
    st.write("Summary : \n\n", result.content[0]["text"])