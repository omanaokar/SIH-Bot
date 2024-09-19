# How to use
### Step 1: Install requirements
Run:
`pip install -r requirements.txt`
or (on VSCode)
`ctrl+shift+p`
Navigate to 'Select Python Interpreter' and create a new virtual environment. When prompted, select `requirements.txt` to download packages.

### Step 2: Update knowledgebase
Run:
`python add_pdfs.py`
When prompted, paste in the paths of the pdfs one by one. Do not paste relative paths. (Follow instructions on the terminal)
**Make sure faiss-index folder is generated**

### Step 3: Run flask server
Run:
`python flask_app.py`

### Step 4: Run streamlit
Run:
`streamlit run streamlit_app.py`


