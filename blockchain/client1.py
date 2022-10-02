import pandas

# Data Input
from client import Client

df = pandas.read_csv('data.csv')
# todo server用完整的数据集训练，client1用一半的数据训练
data = df[:int(len(df) / 2)]

client1 = Client(data, "cli_1")
client1.train()
