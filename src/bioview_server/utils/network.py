import json 
from typing import List 

from bioview_common import validate_message_format, ValidationError, SUPPORTED_COMMANDS

def parse_and_validate_command(data: str, commmand_type = None) -> List:  
    if not data: 
        raise ValidationError("Server returned no response")

    try:
        message = json.loads(data)
    except json.JSONDecodeError:
        raise ValidationError("Invalid JSON format")
    
    # Validate message structure
    required_fields = ['type', 'payload']
    if not validate_message_format(message, required_fields):
        raise ValidationError("Message missing required fields")
    
    # Validate response type
    received_type = message.get('type')
    if commmand_type not in SUPPORTED_COMMANDS:
        raise ValidationError(f"Unsupported response: {received_type}")
    if commmand_type: 
        if commmand_type != received_type:
            raise ValidationError(f"Incorrect response type received: {received_type}. Expected: {response_type}")

    # Validate payload is a dictionary
    received_payload = message.get('payload') 
    if not isinstance(received_payload, dict):
        raise ValidationError(f"payload must be a dict but got {type(received_payload)} instead")
    
    return received_type, received_payload