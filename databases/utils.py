from argon2 import PasswordHasher

pw = PasswordHasher()


def hash(password:str):
    hashed_password = pw.hash(password)
    return hashed_password


def verify(plain_password, hashed_password):
    try:
        is_verified = pw.verify(hashed_password, plain_password)
        return is_verified
    
    except Exception as e:
        return "There is a problem verifying the password"