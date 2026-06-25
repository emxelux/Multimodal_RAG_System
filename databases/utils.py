import hashlib
from argon2 import PasswordHasher

pw = PasswordHasher()


def hash(password:str):
    hashed_password = pw.hash(password)
    return hashed_password


def verify(plain_password, hashed_password):
    try:
        is_verified = pw.verify(hashed_password, plain_password)
        if is_verified:
            return is_verified
        else:
            return "Invalid Credentials"
    
    except Exception as e:
        return "There is a problem With the Credentials"



def hash_pdf(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


