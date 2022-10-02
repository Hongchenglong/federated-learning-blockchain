import pickle
import socket

import numpy

class Client(object):
    def __init__(self, data):
        X = data.drop('charges', axis=1)
        y = numpy.array(data['charges'])
        y = y.reshape((len(y), 1))

        data_inputs = numpy.array(X).T
        self.data_outputs = numpy.array(y).T

        # 数据标准化
        mean = numpy.mean(data_inputs, axis=1, keepdims=True)
        std_dev = numpy.std(data_inputs, axis=1, keepdims=True)
        self.data_inputs = (data_inputs - mean) / std_dev

    # 接收 TCP 数据，数据以字符串形式返回，bufsize 指定要接收的最大数据量。
    def recv(self, soc, buffer_size=1024, recv_timeout=10):
        """
        接收 TCP 数据，数据以字符串形式返回，bufsize 指定要接收的最大数据量。
        Args:
            soc:
            buffer_size:
            recv_timeout:

        Returns:

        """

        received_data = b""
        while str(received_data)[-18:-7] != '-----------':
            try:
                soc.settimeout(recv_timeout)
                received_data += soc.recv(buffer_size)
            except socket.timeout:
                print("A socket.timeout exception occurred because the server "
                      "did not send any data for {recv_timeout} seconds.".format(recv_timeout=recv_timeout))
                return None, 0
            except BaseException as e:
                print("An error occurred while receiving data from the server {msg}.".format(msg=e))
                return None, 0

        try:
            # print(str(received_data)[-18:-7])
            print("All data ({data_len} bytes).".format(data_len=len(received_data)))
            received_data = pickle.loads(received_data)
        except BaseException as e:
            print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
            return None, 0

        return received_data, 1

    def train(self):
        soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        print("Socket Created.\n")

        try:
            soc.connect(("localhost", 10000))
            print("Successful Connection to the Server.\n")
        except BaseException as e:
            print("Error Connecting to the Server: {msg}".format(msg=e))
            soc.close()
            print("Socket Closed.")

        subject = "echo"
        NN_model = None

        while True:
            data = {"subject": subject, "data": NN_model, "mark": "-----------"}
            data_byte = pickle.dumps(data)
            print("data sent to server {}".format(len(data_byte)))
            print("Sending the Model to the Server.\n")
            soc.sendall(data_byte)

            print("Receiving Reply from the Server.")
            received_data, status = self.recv(soc=soc, buffer_size=1024, recv_timeout=10)
            if status == 0:
                print("Nothing Received from the Server.")
                break
            else:
                print(received_data, end="\n\n")

            subject = received_data["subject"]
            if subject == "model":
                NN_model = received_data["data"]
                # print("Architecture of the model {}".format(NN_model.architecture))
                # print("Cost function the model {}".format(NN_model.cost_function))
            elif subject == "done":
                print("Model is trained.")
                break
            else:
                print("Unrecognized message type.")
                break

            NN_model.data = self.data_inputs
            NN_model.labels = self.data_outputs

            # todo
            history = NN_model.train(1000)
            # print(history)
            prediction = NN_model.layers[-1].a  # y_hat
            # print("Predictions from model {predictions}".format(predictions = prediction))
            error = NN_model.calc_accuracy(self.data_inputs, self.data_outputs, "RMSE")  # todo 拿训练集作为测试集？
            print("Error from model(RMSE) {error}".format(error=error))
            # ga_instance.run()

            # ga_instance.plot_result()

            subject = "model"

        soc.close()
        print("Socket Closed.\n")


