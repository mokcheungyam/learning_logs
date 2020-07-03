# 留出验证
num_validation_samples = 10000

np.random.shuffle(data)  # 打乱数据

validation_data = data[:num_validation_samples]  # 定义验证集
data = data[num_validation_samples:]

training_data = data[:]  # 定义训练集

model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

model = get_model()
model.train(np.concatenate([training_data, validation_data]))
test_score = model.evaluate(test_data)



# K折交叉验证
k = 4
num_validation_samples = len(data) // k

np.random.shuffle(data)

validation_scores = []
for fold in range(k):
    validation_data = data[num_validation_samples * fold:
     num_validation_samples * (fold + 1)]
    training_data = data[num_validation_samples * fold] + 
     data[num_validation_samples * (fold + 1):]

    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)

validation_score = np.average(validation_scores)

model = get_model()
model.train(data)
test_score = model.evaluate(test_data)


# 原始模型
from keras import models
from keras improt layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 替代模型
model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, acitvation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))




















































































































