import hashlib
import os 

from bioview_common import SHARED_SECRET

# TODO: Make this better
def generate_challenge(): 
    return os.urandom(16).hex()

def validate_token(challenge, received):
    expected = hashlib.sha256((challenge + SHARED_SECRET).encode()).hexdigest()
    return expected == received
