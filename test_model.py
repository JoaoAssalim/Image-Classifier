from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

image_path = "./Dataset Test/armario/armario (38).jpeg"
image = load_img(image_path, target_size=(128, 128))

image_array = img_to_array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

model = load_model("classifier.h5")

predictions = model.predict(image_array)

predicted_class = np.argmax(predictions)

class_names = ["notebook", "ar condicionado", "Armario", "Cadeira"]
predicted_class_name = class_names[predicted_class]

print("A imagem foi classificada como:", predicted_class_name)

# Modelo 1 (Notebook e Ar condicionado) 94,7%
# 18 acertos / 1 erro
# notec4.jpeg, 


# Modelo 2 (14 Notebook, 5 ar condicionado, 4 cadeira, 5 armario) 87,5%
# 28 acertos / 4 erros
# ar3.jpeg (notebook),  notec2.jpeg (Armario), notec4.jpeg (ar condicionado), notec6.jpeg (Armario), 