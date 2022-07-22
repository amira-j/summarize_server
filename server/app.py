from flask import Flask, render_template, request, send_from_directory
from flask_cors import CORS

import torch
import pdfplumber
import io
import os
from sentence_splitter import SentenceSplitter
from waitress import serve


app = Flask(__name__,template_folder='./build', static_folder='./build/static')
CORS(app)

#tokenizer = torch.load("../pegasus/model/pegasus_token.pt")
#model = torch.load("../torch.load("../pegasus/model/bart_pipeline.pt")
global model
model = torch.load("../pegasus/model/pegasus_pipeline.pt")
global currModel
currModel = 'Pegasus'
max_len=512
try_fac = [3, 2, 1.5, 1]
splitter = SentenceSplitter(language='en')

@app.route("/")
def hello():
	return render_template("index.html")

@app.route("/flask/hello")
def helloFlask():
	return {"status": "Success", "message": "Hello World!!!!"}

@app.route('/summarize/plaintext', methods = ['POST'])
def summarizePlain():
	text = request.form['text']
	sum, chunks = summarize(text, request)
	return {"original": text, "summary":sum, "chunks":chunks, "model":currModel}

@app.route('/summarize/file', methods = ['POST'])
def summarizeFile():
	f = request.files['file']
	print(f)
	text = f.read().decode('utf-8')

	sum, chunks = summarize(text, request)
	return {"original": text, "summary":sum, "chunks":chunks, "model":currModel}


@app.route('/summarize/pdf', methods = ['POST'])
def summerizePDF():
	f = request.files['file']
	text = ''
	with io.BytesIO(f.read()) as open_pdf_file:
		with pdfplumber.open(open_pdf_file) as pdf:
			for p in pdf.pages:
				try:
					text += p.extract_text()
				except:
					pass
	sum, chunks = summarize(text, request)
	return {"original": text, "summary":sum, "chunks":chunks, "model":currModel}

def summarize(text, request):
	params = {}
	if request:
		if request.form['minL']:
			params['min_length'] = int(request.form['minL'])
		if request.form['maxL']:
			params['max_length'] = int(request.form['maxL'])
	checkModel(request)
	temp = model.preprocess(text)
	chunks = 1
	if len(temp['input_ids'][0])> max_len:
		s_list = splitter.split(text)
		temp = model.preprocess(s_list)
		max_sent_l = len(temp['input_ids'][0])
		min_sent_per_line = (max_len//max_sent_l)
		sent_per_line = (max_len*3//max_sent_l)
		while 1:
			try:
				s_list_max = []
				print(sent_per_line)
				for i in range(((len(s_list)-1)//sent_per_line)+1):
					s_list_max.append(''.join(s_list[sent_per_line*i:sent_per_line*(i+1)]))
				text = s_list_max
				temp=model.preprocess(text)
				sent_per_line = (sent_per_line * 3) // 4
				if len(temp['input_ids'][0]) <=  max_len:
					break
				if sent_per_line <= min_sent_per_line:
					sent_per_line = min_sent_per_line
					for i in range(((len(s_list)-1)//sent_per_line)+1):
						s_list_max.append(''.join(s_list[sent_per_line*i:sent_per_line*(i+1)]))
					text = s_list_max
			except:
				print(f'factor to high: {sent_per_line}')
		chunks = len(text)
	summary=model(text, **params)
	s = '\n'.join(map(lambda elem: elem['summary_text'], summary))
	print('summary generated')
	return s, chunks

def checkModel(request):
	global currModel
	global model
	try:
		if request.form['model']=='fast':
			return #take previous model to get fastest time
		elif request.form['model']=='auto':
			return #not implemented yet, should load the model that fits best
		if request.form['model'] != currModel:
			print('different model')
			if request.form['path']:
				model = torch.load(request.form['path'])
			else:
				print(request.form['model'])
			currModel = request.form['model']
	except:
		print('Error while loading'+str(request.form))
serve(app, host="0.0.0.0", port=5545, threads=1)
