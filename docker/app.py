from flask import Flask, render_template, request
import random
import requests
import io
import base64
import matplotlib.image as mpimg
from PIL import Image
import cv2

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def predict_of_image():
    """
        Визуализирует изображение с наложенными боксами
    """
    if request.method == 'POST':
        img = mpimg.imread(request.files['file'], format='jpeg')
        request.files['file'].seek(0)
        res = requests.post("http://localhost:8080/predictions/fasterrcnn",
                            files={'data': request.files['file']}, timeout=5).json()
        color = {}
        for i in res:
            lable, box = list(i.items())[0]
            if not color.get(lable, False):
                color[lable] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color[lable], thickness=2)
            img = cv2.putText(img, lable, ((int(box[0])), int(max(box[1]-15, 0))), cv2.FONT_HERSHEY_SIMPLEX,
                                1, color[lable], 2, cv2.LINE_AA)

        img = Image.fromarray(img)
        my_stringIObytes = io.BytesIO()
        img.save(my_stringIObytes, format='jpeg')
        my_stringIObytes.seek(0)
        image_string = base64.b64encode(my_stringIObytes.read())
        image_string = image_string.decode('utf-8')
        return render_template('show_image.html', filestring=image_string)
    else:
        return render_template('show_image_pass.html')
