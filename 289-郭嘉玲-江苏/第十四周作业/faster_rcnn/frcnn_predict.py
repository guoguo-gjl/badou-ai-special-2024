from frcnn_model import FRCNN
from PIL import Image

frcnn = FRCNN()
try:
    image = Image.open('img/street.jpg')
except:
    print('Open Error! Try again!')
    continue
else:
    r_imgae = frcnn.detect_image(image)
    r_imgae.show()
frcnn.close_session()
