import pickle
import socket
import numpy
import blockchain as bl


class Client(object):

    def __init__(self, data, cli):
        X = data.drop('charges', axis=1)
        y = numpy.array(data['charges'])
        y = y.reshape((len(y), 1))

        # Preparing the NumPy array of the inputs and outputs.
        data_inputs = numpy.array(X).T
        self.data_outputs = numpy.array(y).T

        # 数据标准化
        mean = numpy.mean(data_inputs, axis=1, keepdims=True)
        std_dev = numpy.std(data_inputs, axis=1, keepdims=True)
        self.data_inputs = (data_inputs - mean) / std_dev

        # 区块链
        self.cli = cli
        self.blockchain = bl.Blockchain()
        self.blockchain.create_genesis_block()  # first block of Bitcoin ever mined

    @staticmethod
    def recv(soc, buffer_size=1024, recv_timeout=10):
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
        chain = None

        while True:
            data = {"subject": subject, "data": chain, "mark": "-----------"}
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
                chain = received_data["data"]
                print("Length of chain", len(self.blockchain.chain))
                last_block = self.blockchain.chain[-1]
                print("hash of last block client", last_block.hash)
                for block in chain:
                    new_block = bl.Block(index=block.index,
                                         cli_model=block.cli_model,
                                         fin_model=block.fin_model,
                                         timestamp=block.timestamp,
                                         previous_hash=block.previous_hash,
                                         cli=block.cli,
                                         nonce=block.nonce)
                    print("From ", block.cli)
                    print("previous hash from server", block.previous_hash)
                    proof = block.hash
                    print("hash of this block", proof)
                    self.blockchain = self.blockchain.add_block(new_block, proof)
                    if not self.blockchain:
                        raise Exception("The chain dump is tampered!!")
                # self.blockchain.add_blocks(chain)

                last_block = self.blockchain.chain[-1]
                # print(last_block.fin_model)
                NN_model = last_block.fin_model
                # print("Architecture of the model {}".format(NN_model.architecture))
                # print("Cost function the model {}".format(NN_model.cost_function))
            elif subject == "done":
                print("Model is trained.")
                break
            else:
                print("Unrecognized message type.")
                break

            # ga_instance = prepare_GA(GANN_instance)

            NN_model.data = self.data_inputs
            NN_model.labels = self.data_outputs

            # todo
            history = NN_model.train(1000)
            # print(history)
            prediction = NN_model.layers[-1].a  # y_hat
            # print("Predictions from model {predictions}".format(predictions = prediction))
            error = NN_model.calc_accuracy(self.data_inputs, self.data_outputs, "RMSE")
            print("Error from model(RMSE) {error}".format(error=error))

            subject = "model"
            chain = bl.Block(last_block.index + 1, NN_model, 0, 0, last_block.hash, self.cli)

        soc.close()
        print("Socket Closed.\n")
