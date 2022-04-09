from Crypto.Cipher import DES3
from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import DES3
from Crypto.Cipher.DES3 import adjust_key_parity
from Crypto.Random import get_random_bytes


class TDES_ECB():
    def __init__(self):
        self.BLOCK_SIZE = 8

    def encrypt(self, plainText, key=None):
        if key is None:
            key = adjust_key_parity(get_random_bytes(24))
        des3 = DES3.new(key, DES3.MODE_ECB)
        if type(plainText) is str:
            cipherText = des3.encrypt(
                pad(plainText.encode(encoding='utf-8'), self.BLOCK_SIZE))

        elif type(plainText) is bytes:
            cipherText = des3.encrypt(pad(plainText, self.BLOCK_SIZE))

        return cipherText

    def decrypt(self, plainText, key):
        des3 = DES3.new(key, DES3.MODE_ECB)
        decryptoData = unpad(des3.decrypt(plainText),
                             self.BLOCK_SIZE).decode(encoding='utf-8')
        return decryptoData


class TDES_CBC():
    def __init__(self, iv=None):
        self.BLOCK_SIZE = 8
        self.iv = iv

    def encrypt(self, plainText, key=None):
        if key is None:
            key = adjust_key_parity(get_random_bytes(24))
        if self.iv is None:
            iv = get_random_bytes(8)
        else:
            iv = self.iv

        des3 = DES3.new(key, DES3.MODE_CBC, iv)

        if type(plainText) is str:
            cipherText = des3.encrypt(
                pad(plainText.encode(encoding='utf-8'), self.BLOCK_SIZE))
        elif type(plainText) is bytes:
            cipherText = des3.encrypt(pad(plainText, self.BLOCK_SIZE))

        return cipherText

    def decrypt(self, cipherText, key, iv):
        aes = DES3.new(key, DES3.MODE_CBC, iv)
        decryptoData = unpad(aes.decrypt(cipherText),
                             self.BLOCK_SIZE).decode(encoding='utf-8')
        return decryptoData
