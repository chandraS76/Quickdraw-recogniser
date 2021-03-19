import os
from flask import Flask, request
from flask_cors import CORS
import base64, json
from PIL import Image
import onnxruntime as ort
import numpy


app = Flask(__name__)
cors =CORS(app)

allclasses=['Bird', 'Flower', 'Hand', 'House', 'Pencil', 'Spectacles', 'Spoon', 'Sun', 'Tree', 'Umbrella']

ort_sesh=ort.InferenceSession('model.onnx')

class customdataclass():
	
	def __init__(self, path):

		self.path = path

		
	def func(self):
		image = Image.open(self.path).convert('RGBA')
		alpha = image.split()[-1]
		return alpha


		

@app.route('/upload', methods=['POST'])
def upload_canvas():
	
	
	data= json.loads(request.data.decode('utf-8'))
	imagedat= data['image'].split(',')[1].encode('utf-8')
	open(r"D:\Code\Autonise\Flaskfile.png",'wb').close()
	
	with open(r"D:\Code\Autonise\Flask\Flaskfile.png", 'wb') as fh:
		fh.write(base64.decodebytes(imagedat))
	
	global model
	
	data=customdataclass('Flaskfile.png').func()
	data = numpy.array(data, dtype=numpy.float32).reshape((1,1,384,384))
	print(data.shape)
	output= ort_sesh.run(None, {'data': data})[0].argmax()

	return allclasses[output]