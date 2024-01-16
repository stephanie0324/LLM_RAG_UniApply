<h1 align="center">UniApply </h1>

<p align="center">
    <a href="https://github.com/FlagOpen/FlagEmbedding">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-lightblue">
    </a>
    <a href="https://github.com/stephanie0324/LLM_RAG_UniApply/stargazers">
        <img alt="Build" src="https://img.shields.io/github/stars/stephanie0324/LLM_RAG_UniApply.svg?color=yellow&style=flat&label=Stars&logoColor=white">
    </a>
    <a href="https://github.com/stephanie0324/LLM_RAG_UniApply/forks">
        <img alt="badge" src="https://img.shields.io/github/forks/stephanie0324/LLM_RAG_UniApply.svg?style=flat&label=Forks">
    </a>
    <a href="https://github.com/stephanie0324/LLM_RAG_UniApply/issues">
        <img alt="badge" src="https://img.shields.io/github/issues/stephanie0324/LLM_RAG_UniApply.svg?style=flat&label=Issues&color=lightpink">
    </a>
</p>
<div align="center">
    <img src="./img/UniApplyLogo.png" width=250, height=150>
    <p>
    This is a Uni/Colledge Application F&Q Chatbot through RAG implementation.
    <br>  
    Ask freely and get the most accurate information with link.
    <br>
    <h5 align="center">
    <p>
        <a href=#about-the-project>About</a> |
        <a href=#new-updates>News</a> |
        <a href="#project-lists">Projects</a> |
        <a href=#usage>Usage</a> |
        <a href="#roadmap">Roadmap</a> |
        <a href="#contributing">Contributing</a> |
        <a href="#license">License</a> 
    </p>
  </h5>
  </p>
</div>

# About the project
<div align="center">
    <img  src = "./img/app_interface.png" width = 500/>
</div>

This is an implementation of the Retrieval-Augmented Generation (RAG) model by [Langchain](https://www.langchain.com/), with application F&Q collected from universities around the world. Providing accurate information and link for your reference, also allows user to ask free form questions.
<p align="center">
<img src="./img/gpt_result.png" alt="drawing" width="200" height="140"/><img src="./img/gpt_rag_result.png" alt="drawing" width="300"/>
<br> Comparison on GPT-3.5 and GPT3.5 with RAG </br>
</p>

# Usage
### Things to install
```bash
!pip install openai
!pip install tiktoken
!pip install langchain
!pip install faiss-cpu
!pip install text_generation
!pip install django
```
### ENV VAR setting
```bash
export "OPENAI_API_KEY" = 'YOUR_OPENAI_KEY'
export "DJANGO_SECRET_KEY" = 'YOUR_DJANGO_SECRET_KEY'
```
### Run
```
cd UniApply_Chatbot
python manage.py runserver 8003 -- you can change port 
```


