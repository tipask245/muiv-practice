# muiv-practice
TensorFlow neural network

Сверточная модель нейросети, обученная на датасете MNIST и определяющая рукописные цифры.

## Установка зависимостей
```bash
pip install -r requirements.txt
```

### MAIN.PY
В файле main.py расположена структура модели.

### NEW_TRAIN.PY
В этом файле можно загрузить уже обученную модель для продолжения обучения.

### PREDICT.PY
Файл для тестирования модели
```python
loaded_model = tf.keras.models.load_model('путь и название модели')
image = load_img('путь и название картинки')
```