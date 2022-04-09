from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes


class AES_ECB():
    def __init__(self):
        self.BLOCK_SIZE = 16

    def encrypt(self, plainText, key=None):

        if key is None:
            key = get_random_bytes(16)
        aes = AES.new(key, AES.MODE_ECB)

        if type(plainText) is str:
            cipherText = aes.encrypt(
                pad(plainText.encode(encoding='utf-8'), self.BLOCK_SIZE))

        elif type(plainText) is bytes:
            cipherText = aes.encrypt(pad(plainText, self.BLOCK_SIZE))

        return cipherText

    def decrypt(self, cipherText, key):

        aes = AES.new(key, AES.MODE_ECB)
        decryptoData = unpad(aes.decrypt(cipherText),
                             self.BLOCK_SIZE).decode(encoding='utf-8')
        return decryptoData


class AES_CBC():
    def __init__(self, iv=None):
        self.BLOCK_SIZE = 16
        self.iv = iv

    def encrypt(self, plainText, key=None):
        if key is None:
            key = get_random_bytes(16)
        if self.iv is None:
            iv = get_random_bytes(16)
        else:
            iv = self.iv

        aes = AES.new(key, AES.MODE_CBC, iv)

        if type(plainText) is str:
            cipherText = aes.encrypt(
                pad(plainText.encoding('utf-8'), self.BLOCK_SIZE))
        elif type(plainText) is bytes:
            cipherText = iv + aes.encrypt(pad(plainText, self.BLOCK_SIZE))

        return cipherText

    def decrypt(self, cipherText, key, iv):
        aes = AES.new(key, AES.MODE_CBC, iv)
        decryptoData = unpad(aes.decrypt(cipherText),
                             self.BLOCK_SIZE).decode(encoding='utf-8')
        return decryptoData
