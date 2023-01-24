#from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
"""from tensorflow.keras.utils import load_img
from keras.models import model_from_json
import numpy as np

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

print("Loaded model from disk")
label = ["after", "before"]
path2 = "F:/Assigments/Accident/archive (1)/TestDataset/frame1_49.jpg"
test_image = load_img(path2, target_size=(128, 128))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = loaded_model.predict(test_image)
label2 = label[result.argmax()]
print(label2)"""

from tensorflow.keras.utils import load_img
from keras.models import model_from_json
import numpy as np

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

print("Loaded model from disk")
label = ["after", "before"]
path2 = "F:/Assigments/Accident/archive (1)/carwinsplash.mp4"
test_image = load_img(path2, target_size=(128, 128))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = loaded_model.predict(test_image)
label2 = label[result.argmax()]
print(label2)
