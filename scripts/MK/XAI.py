import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
path = os.getcwd()

model2 = keras.models.load_model("scripts/Conv.h5")
imagenet = keras.models.load_model("scripts/imagenet.h5")


path = os.path.join(path, 'dataset')

# declare train_path
path_train = os.path.join(path, 'Train')

# declare test_path
path_test = os.path.join(path, 'Test')

train_datagen = ImageDataGenerator(rescale=1.0 / 255
                                   )
explainer = lime_image.LimeImageExplainer()

validation_generator = train_datagen.flow_from_directory(path_test,
                                                              batch_size=10,
                                                              target_size=(150, 100))

validation_generator = train_datagen.flow_from_directory(path_test,
                                                              batch_size=10,
                                                              target_size=(224, 224))

img = next(validation_generator)[0][0].reshape(224, 224, 3)
plt.imshow(img.reshape(224, 224, 3))


explanation = explainer.explain_instance(img.astype('double'),
                                         imagenet.predict,
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000)


temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)

plt.imshow(mark_boundaries(temp, mask))

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=8, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))


temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False, min_weight=0.2)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

model.predict(img.reshape(1, 150, 100, 3))


y_proba = model.predict(img.reshape(1, 150, 100, 3))
y_classes = keras.utils.probas_to_classes(y_proba)

model.predict_classes()

ind =  explanation.top_labels[0]

#Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

#Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()

