import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentence = ["Eu estou feliz em encontrar meus amigos. Estamos planejando ir a uma festa.", 
            "Eu tive um dia ruim na escola. Eu me machuquei jogando futebol."]
# Tokenização

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(sentence)

#Crie um dicionário chamado word_index

word_index = tokenizer.word_index

sequence = tokenizer.texts_to_sequences(sentence)

print(sequence[0:2])

# Preenchendo a sequência

padded = pad_sequences(sequence, maxlen=100, 
                      padding='post', truncating='post')

print(padded[0:2])

# Defina o modelo usando um arquivo .h5

model=tensorflow.keras.models.load_model('Text_Emotion.h5')


# Teste o modelo

result=model.predict(padded)

print(result)

# Imprima o resultado

predict_class = np.argmax(result, axis=1)

print(predict_class)

# {"anger": 0, "fear": 1, "joy": 2, "love": 3, "sadness": 4, "surprise": 5}
