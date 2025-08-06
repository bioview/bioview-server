import queue
from bioview_common import ConnectionStatus, Response

def log_event(response_queue, level, message):
    if level == "error":
        msg_type = Response.ERROR
    elif level == "warning":
        msg_type = Response.WARNING
    elif level == "info":
        msg_type = Response.INFO
    else:
        msg_type = Response.DEBUG

    response = {
        'type': msg_type.value,
        'payload': {
            'message': message
        }
    }
    
    try: 
        response_queue.put_nowait(response)
    except queue.Full: 
        print('Unable to add to response queue as it is full.')

def connection_state_changed(response_queue, device_id, status: ConnectionStatus):
    response = {
        'type': Response.DEVICE_STATUS_CHANGED,
        'payload': {
            'device_id': device_id, 
            'status': status.value
        }
    }
    
    try: 
        response_queue.put_nowait(response)
    except queue.Full: 
        print('Unable to add to response queue as it is full.')