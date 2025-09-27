import json
from typing import List

from bioview_common import (
    Response, 
    SUPPORTED_COMMANDS, 
    SUPPORTED_RESPONSES,
    MAX_BUFFER_SIZE, 
    ValidationError,
    log_print
)

def send_response(sock, response, params, logger = None, buffer_size: int = MAX_BUFFER_SIZE): 
    if not isinstance(response, Response) or response.name not in SUPPORTED_RESPONSES:
        log_print(logger, 'error', f'Invalid response: {response}')
        return None 
    
    try:
        response_dict = {
            'type': response.name,
            'payload': params or {}
        }
        response_json = json.dumps(response_dict).encode("utf-8")
        sock.send(response_json)
    except Exception as e: 
        msg = f'Error occurred while sending response: {e}'
        log_print(logger, 'error', msg, fallback=True)

def parse_and_validate_command(data: str) -> List:
    if not data:
        raise ValidationError("Client provided an empty command")

    try:
        message = json.loads(data.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise ValidationError("Invalid JSON format") from e

    # Validate command type
    cmd_type = message.get("type", None)
    if not cmd_type or cmd_type not in SUPPORTED_COMMANDS: 
        raise ValidationError(f"Invalid command: {cmd_type}")

    # Validate payload is a dictionary
    cmd_payload = message.get("payload")
    if not isinstance(cmd_payload, dict):
        raise ValidationError(
            f"payload must be a dict but got {type(cmd_payload)} instead"
        )

    return cmd_type, cmd_payload