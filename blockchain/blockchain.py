from hashlib import sha256
import json


class Block:
    def __init__(self, index, cli_model, fin_model, timestamp, previous_hash, cli, nonce=0):
        """
        todo
        Args:
            index:
            cli_model:
            fin_model:
            timestamp: 时间戳
            previous_hash: 前一个区块的哈希
            cli:
            nonce:
        """
        self.index = index
        self.cli_model = cli_model
        self.fin_model = fin_model
        self.cli = cli
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = nonce

    def compute_hash(self):
        """
        A function that return the hash of the block contents.
        """
        cli_model = self.cli_model
        temp = [self.index, self.cli, self.timestamp, self.previous_hash, self.nonce]
        if cli_model != 0:
            for i in range(len(cli_model.layers)):
                temp.append(cli_model.layers[i].W.tolist())
                temp.append(cli_model.layers[i].b.tolist())

        block_bytes = json.dumps(temp)
        return sha256(block_bytes.encode()).hexdigest()


class Blockchain:
    # difficulty of our PoW algorithm
    difficulty = 1

    def __init__(self):
        self.unconfirmed_transactions = []
        self.chain = []

    def create_genesis_block(self):
        """
        创世块
        A function to generate genesis block and appends it to the chain.
        The block has index 0, previous_hash as 0, and a valid hash.
        """
        genesis_block = Block(0, 0, 0, 0, 0, "0")
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)

    @property
    def last_block(self):
        return self.chain[-1]

    def add_block(self, block, proof):
        """
        A function that adds the block to the chain after verification.
        Verification includes:
        * Checking if the proof is valid.
        * The previous_hash referred in the block and the hash of the latest block
          in the chain match.
        """
        previous_hash = self.last_block.hash
        if previous_hash != block.previous_hash:
            return False
        if not Blockchain.is_valid_proof(block, proof):
            return False
        # print("reached")
        block.hash = proof
        self.chain.append(block)
        return self

    def add_blocks(self, chain_dump):
        """
        Add the blocks coming from server after verifying them.
        """
        for idx, block_data in enumerate(chain_dump):
            # if idx == 0:
            #     continue  # skip genesis block
            block = Block(block_data.index,
                          block_data.cli_model,
                          block_data.fin_model,
                          block_data.cli,
                          block_data.timestamp,
                          block_data.previous_hash,
                          block_data.nonce)
            proof = block_data.hash
            # 验证proof是否是block的哈希后，再添加到链上
            added = self.add_block(block, proof)
            if not added:
                raise Exception("The chain dump is tampered!!")

    @staticmethod
    def proof_of_work(block):
        """
        Function that tries different values of nonce to get a hash
        that satisfies our difficulty criteria.
        """
        block.nonce = 0

        computed_hash = block.compute_hash()
        while not computed_hash.startswith('0' * Blockchain.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()

        return computed_hash

    def add_new_transaction(self, transaction):
        self.unconfirmed_transactions.append(transaction)

    @classmethod
    def is_valid_proof(cls, block, block_hash):
        """
        Check if block_hash is valid hash of block
        and satisfies the difficulty criteria.
        """
        print("hash calculated", block.compute_hash())

        return (block_hash.startswith('0' * Blockchain.difficulty)
                and block_hash == block.compute_hash())

