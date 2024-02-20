# 1.	DATASET
El dataset Fashion MNIST se trata de un conjunto de datos de 60.000 imágenes en escala de grises de 28x28 de 10 categorías de moda, junto con un conjunto de prueba de 10.000 imágenes. Las clases son:
Label	Class
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot

# 2.	PROCEDIMIENTO
Primero se realizó la importación y cargar del set de datos de MNIST directamente de TensorFlow. A continuación, se exploró el formato de el set de datos antes de entrenar el modelo. Se constató que hay 60,000 imágenes en el set de entrenamiento y 10,000 imágenes en el set de pruebas, con cada imagen representada por pixeles de 28x28.

Se redimensionó cada una de las imágenes a (28, 28, 1) tanto en entrenamiento como en test, el formato ideal para introducirlo al modelo.

Se normalizó los datos en el conjunto de entrenamiento x_train y test x_test, para mejorar la convergencia del modelo y acelerar el proceso de entrenamiento.

Antes de aplicar el modelo, se verifica que el set de datos este en el formato adecuado, desplegando las primeras 25 imágenes del conjunto de entrenamiento con el nombre de cada clase debajo de cada imagen.

Tanto para el conjunto de entrenamiento y test las etiquetas se binarizan. Cada etiqueta será representada como un vector binario donde una posición tendrá un valor de 1 para indicar la clase y todas las demás posiciones tendrán un valor de 0.


Se definió un modelo de red neuronal convolucional (CNN) diseñado para tareas de clasificación de imágenes, compuesto por capas convolucionales, capas de agrupación máxima, capas de aplanamiento, capas densas (totalmente conectadas) y capas de abandono (dropout) 

Capa de entrada (Input): Toma una forma de entrada especificada por input_shape (28, 28, 1).

Capa convolucional (Conv2D): 
Utiliza 32 filtros de tamaño 2x2. Usa una función de activación ReLU.
Usa un relleno ('same') para mantener la misma dimensión espacial de salida que la de entrada.
Usa un paso (strides) de 2x2 para reducir la dimensión espacial.

Capa de agrupación máxima (MaxPooling2D):
Reduce la dimensionalidad espacial de la salida de la capa convolucional.
Utiliza un pool size de 2x2.
Usa un relleno ('same') para mantener la misma dimensión espacial de salida que la de entrada.
Usa un paso (strides) de 1x1.

Capa convolucional (Conv2D):
Utiliza 64 filtros de tamaño 2x2.
Usa una función de activación ReLU.
Usa un relleno ('same') para mantener la misma dimensión espacial de salida que la de entrada.
Usa un paso (strides) de 2x2 para reducir la dimensión espacial.

Otra capa de agrupación máxima (MaxPooling2D):
Reduce la dimensionalidad espacial de la salida de la segunda capa convolucional.
Utiliza un pool size de 2x2.
Usa un relleno ('same') para mantener la misma dimensión espacial de salida que la de entrada.
Usa un paso (strides) de 1x1.

Capa de aplanamiento (Flatten):
Convierte los mapas de características 2D en un vector unidimensional para alimentar las capas densas.

Capa densa (Dense):
Tiene 256 neuronas.
Utiliza una función de activación ReLU.

Capa de abandono (Dropout):
Descarta aleatoriamente el 50% de las unidades de la capa anterior durante el entrenamiento para evitar el sobreajuste.

Otra capa densa (Dense):
Tiene 128 neuronas.
Utiliza una función de activación ReLU.

Otra capa de abandono (Dropout):
Descarta aleatoriamente el 50% de las unidades de la capa anterior durante el entrenamiento.

Capa de salida (Dense):
Tiene 10 neuronas (correspondientes a las 10 clases en el conjunto de datos de salida).
Utiliza una función de activación softmax para generar una distribución de probabilidad sobre las clases.

Se configura el modelo para entrenar durante 50 épocas, utilizando lotes de 128 muestras a la vez, utilizando la función de pérdida de entropía cruzada categórica y el optimizador Adam, mientras se monitorea la precisión durante el entrenamiento.

Se entrenó el modelo durante el número especificado de épocas (epochs), utilizando los datos de entrenamiento y datos de validación para evaluar el rendimiento del modelo. Durante el entrenamiento, se optimizan los pesos del modelo para minimizar la función de pérdida especificada durante la compilación (categorical_crossentropy en este caso) utilizando el optimizador Adam.


# 3.	RESULTADOS
Se calculó la pérdida y la precisión del modelo en el conjunto de datos de prueba y entrenamiento.

El modelo obtuvo los siguientes resultados en el conjunto de datos de prueba:

Pérdida (Loss): 0.42146244645118713
Precisión (Accuracy): 0.9059000015258789
Esto significa que, en el conjunto de datos de prueba, el modelo tiene una pérdida promedio de aproximadamente 0.421 y una precisión de clasificación de aproximadamente el 90.59%. Una pérdida baja y una precisión alta indican un buen rendimiento del modelo en la tarea de clasificación.

Para el conjunto de datos de entrenamiento se obtuvo lo siguiente:

Pérdida (Loss): 0.04058331251144409
Precisión (Accuracy): 0.9854375123977661

Esto significa que, en el conjunto de datos de entrenamiento, el modelo tiene una pérdida promedio de aproximadamente 0.0406 y una precisión de clasificación de aproximadamente el 98.54%. Una pérdida baja y una precisión alta en el conjunto de entrenamiento sugieren que el modelo se está ajustando bien a los datos de entrenamiento y es capaz de clasificar correctamente la mayoría de las muestras en este conjunto de datos.


Se realizó las predicciones utilizando el modelo entrenado en el conjunto de datos de prueba x_test, donde se observa lo siguiente: 
Clase 0 (T-shirt/top) el modelo tiene un rendimiento con una precisión global del 87%
Clase 1 (Trouser) el modelo tiene un rendimiento bastante bueno, con una precisión global del 100%
Clase 2 (Pullover) el modelo tiene un rendimiento con una precisión global del 84%
Clase 3(Dress) el modelo tiene un rendimiento con una precisión global del 89%
Clase 4 (Coat) el modelo tiene un rendimiento con una precisión global del 87%
Clase 5 (Sandal) el modelo tiene un rendimiento con una precisión global del 97%
Clase 6 (Shirt) el modelo tiene un rendimiento con una precisión global del 74%
Clase 7 (Sneaker) el modelo tiene un rendimiento con una precisión global del 94%
Clase 8 (Bag) el modelo tiene un rendimiento con una precisión global del 98%
Clase 9 (Ankle boot) el modelo tiene un rendimiento con una precisión global del 98%

Estos resultados muestran que el modelo tiene un rendimiento bastante bueno, con una precisión global del 91% en el conjunto de datos de prueba. 

Al realizar la Matriz de confusión para el conjunto de datos de test, se aprecia en la diagonal principal el número de predicciones correctas para cada clase. 
Clase 0 (T-shirt/top) se tiene 822 verdaderos positivos 
Clase 1 (Trouser) 980 verdaderos positivos 
Clase 2 (Pullover) 864 verdaderos positivos
Clase 3(Dress) 924 verdaderos positivos 
Clase 4 (Coat) 840 verdaderos positivos
Clase 5 (Sandal) 980 verdaderos positivos
Clase 6 (Shirt) 752 verdaderos positivos
Clase 7 (Sneaker) 974 verdaderos positivos
Clase 8 (Bag) 972 verdaderos positivos
Clase 9 (Ankle boot) 951 verdaderos positivos

Se observa que se tiene la menor precisión en la clase 6 y el número de muestras que fueron clasificadas incorrectamente es mayor que las otras clases.


# 4.	CONCLUSIONES
Se observa que las predicciones hechas han sido bastante buenas, con un total de 9059 predicciones correctas respecto 941 erróneas.

Se aplicó técnicas de reducción de la dimensionalidad (MaxPooling2D), capas de regularización (Dropouts), 50 épocas y un batch size de 128, lo que ha permitió no tener overfitting.

El modelo entrenado ha sido bastante bueno para la predicción, ha obtenido una precisión de 0.9059 para los datos test.


