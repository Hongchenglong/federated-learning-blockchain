import pickle
import socket
import threading
import time

import numpy
import pandas

import backprop as bp


class SocketThread(threading.Thread):
    """
    run -> recv -> reply -> model_averaging
    """

    def __init__(self, connection, client_info, buffer_size=1024, recv_timeout=5):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size  # 接收的最大数据量
        self.recv_timeout = recv_timeout
        # self.lock = threading.Lock()

    def recv(self):
        """
        接收字节流数据，判断是否非空，若非空则解码数据
        """
        received_data = b""
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data += data

                # Nothing received from the client.
                if data == b'':
                    received_data = b""
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute,
                    # return with status 0 to close the connection.
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0  # 0 means the connection is no longer active and it should be closed.
                elif str(received_data)[-18:-7] == '-----------':
                    # print(str(received_data)[-19:-8])
                    # print("All data ({data_len} bytes) Received from {client_info}."
                    # .format(client_info=self.client_info, data_len=len(received_data)))

                    if len(received_data) > 0:
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1
                        except BaseException as e:
                            print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0
                else:
                    # In case data are received from the client,
                    # update the recv_start_time to the current time to reset the timeout counter.
                    self.recv_start_time = time.time()
            except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def model_averaging(self, model, other_model):
        """
        平均聚合两个模型

        Args:
            model: server的模型
            other_model: client的模型

        Returns: 模型

        """
        for i in range(len(model.layers)):
            W_a = model.layers[i].W
            W_b = other_model.layers[i].W
            b_a = model.layers[i].b
            b_b = other_model.layers[i].b
            model.layers[i].W = (W_a + W_b) / 2
            model.layers[i].b = (b_a + b_b) / 2

        return model

    def reply(self, received_data):
        """
        判断subject是echo或model，
        echo：回传默认或已有模型
        model：聚合模型
        并根据误差判断回传的model类型

        Args:
            received_data: 接收的client数据

        Returns:

        """
        # self.lock.acquire()
        global NN_model, data_inputs, data_outputs, model, counter, response
        if type(received_data) is dict:
            if ("data" in received_data.keys()) and ("subject" in received_data.keys()):
                subject = received_data["subject"]
                if subject == "echo":
                    if model is None:
                        # 如果是第一次调用，回传默认的模型
                        data = {"subject": "model", "data": NN_model, "mark": "-----------"}
                    else:
                        # todo server的data_inputs和client的数据集一致
                        # model是聚合后的模型
                        # model不为空，评估误差是否为0
                        model.data = data_inputs
                        model.labels = data_outputs
                        model.forward_pass()
                        error = model.calc_accuracy(data_inputs, data_outputs, "RMSE")

                        # predictions = model.layers[-1].a
                        # error = numpy.sum(numpy.abs(predictions - data_outputs))/data_outputs.shape[0]
                        # In case a client sent a model to the server despite that the model error is 0.0.
                        # In this case, no need to make changes in the model.
                        if error == 0:
                            # done表示结束
                            data = {"subject": "done", "data": model, "mark": "-----------"}
                        else:
                            # model表示继续训练
                            data = {"subject": "model", "data": model, "mark": "-----------"}
                    try:
                        response = pickle.dumps(data)
                    except BaseException as e:
                        print("Error Encoding the Message: {msg}.\n".format(msg=e))
                elif subject == "model":
                    try:
                        best_model = received_data["data"]
                        if model is None:
                            model = best_model
                            print(model)
                        else:
                            # todo 为什么要计算model误差
                            # 模型不为空，训练新模型，与传输的模型聚合
                            # print("shape of data {input}".format(input=data_inputs.shape))
                            model.data = data_inputs
                            model.labels = data_outputs
                            model.forward_pass()
                            error = model.calc_accuracy(data_inputs, data_outputs, "RMSE")

                            # In case a client sent a model to the server despite that the model error is 0.0.
                            # In this case, no need to make changes in the model.
                            if error <= 0.15:
                                data = {"subject": "done", "data": None, "mark": "-----------"}
                                response = pickle.dumps(data)
                                # print("Error in total {}".format(error))
                                # return
                            else:
                                # todo 如果现有模型的误差大于0.15，那么就跟现有模型聚合，以提升性能
                                model = self.model_averaging(model, best_model)

                        # 判断回传的模型误差
                        model.data = data_inputs
                        model.labels = data_outputs
                        model.forward_pass()
                        error = model.calc_accuracy(data_inputs, data_outputs, "MAE")

                        counter += 1
                        print("Error(RMSE) from {info} = {error}\ncounter {counter}"
                              .format(error=error, info=self.client_info, counter=counter))

                        if error >= 0.15:
                            # model，继续训练
                            data = {"subject": "model", "data": model, "mark": "-----------"}
                            response = pickle.dumps(data)
                            print("sent", data)
                            print("data_sent", len(response))
                        else:
                            # done 完成
                            data = {"subject": "done", "data": None, "mark": "-----------"}
                            response = pickle.dumps(data)

                    except BaseException as e:
                        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                else:
                    response = pickle.dumps("Response from the Server")

                try:
                    # print("Len of data sent {}".format(len(response)))
                    self.connection.sendall(response)
                except BaseException as e:
                    print("Error Sending Data to the Client: {msg}.\n".format(msg=e))

            else:
                print("The received dictionary from the client must have the 'subject' and 'data' keys available. "
                      "The existing keys are {d_keys}.".format(d_keys=received_data.keys()))
        else:
            print("A dictionary is expected to be received from the client but {d_type} received.".format(
                d_type=type(received_data)))
        # self.lock.release()

    def run(self):
        # print("Running a Thread for the Connection with {client_info}.".format(client_info=self.client_info))

        # This while loop allows the server to wait for the client to send data more than once within the same connection.
        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = "Waiting to Receive Data Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT".format(
                year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour,
                minute=time_struct.tm_min, second=time_struct.tm_sec)
            print(date_time)
            # todo
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print(
                    "Connection Closed with {client_info} either due to inactivity for {recv_timeout} seconds or due "
                    "to an error.".format(
                        client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
                break

            # print(received_data)
            sema.acquire()
            # todo
            self.reply(received_data)
            sema.release()


model = None
counter = 0
# https://towardsdatascience.com/visualising-data-with-seaborn-who-pays-more-for-health-insurance-200d01892ba5
# 数据集 https://www.kaggle.com/datasets/mirichoi0218/insurance
# 数据集分析 https://github.com/chongjason914/seaborn-tutorial/blob/main/insurance-data-visualisation.ipynb
df = pandas.read_csv('data.csv')
# X去掉charges列，y取charges列
X = df.drop('charges', axis=1)
y = df['charges']
y = numpy.array(y)
y = y.reshape((len(y), 1))

# Preparing the NumPy array of the inputs.
data_inputs = numpy.array(X)
# Preparing the NumPy array of the outputs.
data_outputs = y
# 转置
data_inputs = data_inputs.T
data_outputs = data_outputs.T
# https://www.cnblogs.com/peixu/articles/13393958.html
mean = numpy.mean(data_inputs, axis=1, keepdims=True)  # 均值
std_dev = numpy.std(data_inputs, axis=1, keepdims=True)  # 标准差
# 数据标准化：先减去均值，再除以方差
# 减去均值是为了突出差异；除以方差，使网络关注整个图像的变化
data_inputs = (data_inputs - mean) / std_dev

num_inputs = 12
num_classes = 1
sema = threading.Semaphore()

description = [{"num_nodes": 12, "activation": "relu"},
               {"num_nodes": 1, "activation": "relu"}]

NN_model = bp.NeuralNetwork(description, num_inputs, "mean_squared", data_inputs, data_outputs, learning_rate=0.001)

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created.\n")

# Timeout after which the socket will be closed.
# soc.settimeout(5)

soc.bind(("localhost", 10000))
print("Socket Bound to IPv4 Address & Port Number.\n")

# 开始 TCP 监听。backlog 指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为 1，大部分应用程序设为 5 就可以了。
soc.listen(1)
print("Socket is Listening for Connections ....\n")

all_data = b""
while True:
    try:
        # 被动接受TCP客户端连接,(阻塞式)等待连接的到来
        connection, client_info = soc.accept()
        # print("New Connection from {client_info}.".format(client_info=client_info))
        socket_thread = SocketThread(connection=connection,
                                     client_info=client_info,
                                     buffer_size=1024,
                                     recv_timeout=10)
        socket_thread.start()
    except:
        soc.close()
        print("(Timeout) Socket Closed Because no Connections Received.\n")
        break
