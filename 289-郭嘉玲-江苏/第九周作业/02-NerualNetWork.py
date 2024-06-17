import numpy
import scipy.special

class NeuralNetWork:
    # 网络初始化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        # numpy.random.normal()，生成符合正态分布的随机数，0.0是均值，pow(self.hnodes, -0.5)是标准差，取self.hnodes的负平方根
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))
        # 对输入x应用激活函数
        self.activation_function = lambda x:scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # 计算误差
        outputs_errors = targets - final_outputs
        # 【手推一遍】
        hidden_errors = numpy.dot(self.who.T, outputs_errors*final_outputs*(1-final_outputs))
        self.who += self.lr * numpy.dot((outputs_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))
        pass

    def query(self, inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

# 初始化网络
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 读入训练数据
training_data_file = open('dataset/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# 模型训练
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        # 【不理解这里】
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
test_data_file = open('dataset/mnnist_test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print('该图片对应的数字是', correct_number)
    inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print('网络认为图片的数字是：', label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
scores_array = numpy.asfarray(scores)
print('performance = ', scores_array.sum() / scores_array.size)