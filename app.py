
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import ast
import json
import re

app = Flask(__name__)
CORS(app)

# Configure paths for Tesseract and uploads
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_document_text(file_path, file_type):
    text = ""
    if file_type == "application/pdf":
        pdf_reader = PdfReader(file_path)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text() or ""
            text += extracted_text
            print(f"Extracted from PDF page: {extracted_text[:50]}")  
    elif file_type == "image/png":
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        print(f"Extracted from image: {text[:50]}") 
    return text

def extract_code_from_response(response):
    start = response.find('const data = [')
    end = response.find('];', start) + 2
    return response[start:end]


def safe_parse(input_str):
    input_str = input_str.strip()
    data = {}

    try:
        data = json.loads(input_str)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        input_str = input_str.strip('{}').strip()
        if ':' in input_str:
            pairs = input_str.split(',')
            cleaned_pairs = []
            for pair in pairs:
                key, value = map(str.strip, pair.split(':'))
                if value.startswith('{') and value.endswith('}'):
                    value = value.strip('{}')
                    value_list = value.split(';') 
                    cleaned_value = '[' + ', '.join(value_list) + ']'
                    cleaned_pairs.append(f"'{key}': {cleaned_value}")
                elif value.isdigit():
                    cleaned_pairs.append(f"'{key}': {value}")
                elif value.startswith("'") and value.endswith("'"):
                    cleaned_pairs.append(f"'{key}': {value}")
                else:
                    cleaned_pairs.append(f"'{key}': '{value}'")
            cleaned_input = '{' + ', '.join(cleaned_pairs) + '}'
        else:
            key, value = map(str.strip, input_str.split(','))
            if value.isdigit():
                cleaned_input = f"{{'{key}': {value}}}"
            else:
                cleaned_input = f"{{'{key}': '{value}'}}"
        try:
            data = ast.literal_eval(cleaned_input)
        except Exception as e:
            print(f"Failed to parse text input: {str(e)}")
            data = {}

    return data

def transform_to_array(data):
    result = []
    if isinstance(data, list):
        for item in data:
            for key, value in item.items():
                result.append({'title': key, 'value': value})
    elif data:
        for key, value in data.items():
            result.append({'title': key, 'value': value})
    return result


def safe_parse(input_str):
    input_str = input_str.strip()

    if input_str.startswith('const data ='):
        input_str = input_str[12:].strip()

    data = {}

    try:
        fixed_input_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', input_str)
        fixed_input_str = re.sub(r',\s*}', '}', fixed_input_str)
        fixed_input_str = re.sub(r',\s*\]', ']', fixed_input_str)
        data = json.loads(fixed_input_str)
    except json.JSONDecodeError:
        try:
            data = {}
            input_str = re.sub(r'^[{[]\s*|\s*[]}]\s*$', '', input_str)
            pairs = input_str.split(',')
            for pair in pairs:
                if ':' in pair:
                    key, value = map(str.strip, pair.split(':', 1))
                    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                        value = value[1:-1]  
                    elif value.isdigit(): 
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit(): 
                        value = float(value)
                    data[key] = value
                else:
                    raise ValueError("Unable to parse the input string correctly.")
        except Exception as e:
            print(f"Error parsing input: {str(e)}")
            data = {}

    return data

def transform_to_array(data):
    result = []
    if isinstance(data, list):
        for item in data:
            for key, value in item.items():
                result.append({'title': key, 'value': value})
    elif data:
        for key, value in data.items():
            result.append({'title': key, 'value': value})
    return result


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            file_type = file.content_type
            document_text = get_document_text(file_path, file_type)
            
            text_chunks = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=500).split_text(document_text)
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            llm = ChatOpenAI()
            memory = ConversationBufferMemory(memory_key='chat_history')
            conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
            question = "Code me table data in 1 array of objects called data in JavaScript."
            response = conversation_chain({'question': question})
            answer = response.get('answer')
            print(answer)
            parsed_data = safe_parse(answer)
            output = transform_to_array(parsed_data)
            print('const data =', json.dumps(output, indent=4))
            return jsonify({'data': output}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)


