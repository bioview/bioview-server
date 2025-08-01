from .config import MultiUsrpConfiguration, UsrpConfiguration
from .device import MultiUsrpDevice

def get_device_object(device_name, config, resp_queue, data_queue, save=False, save_path=None):
    return MultiUsrpDevice(
        device_name=device_name, config=config, resp_queue=resp_queue, data_queue=data_queue, save=save, save_path=save_path
    )

def discover_devices(): 
    devices = [] 
    try: 
        # Check if uhd imported, and if not, import it 
        import sys 
        if 'uhd' in sys.modules:
            uhd = sys.modules['uhd']
        else: 
            import uhd 
            
        device_list = uhd.find("")
    
        for device in device_list: 
            device_dict = dict(device)
            device_dict['handler_type'] = 'usrp'
            devices.append(device_dict)
    
    except Exception as e: 
        print(f'Error occured in UHD device discovery: {e}')
        
    return devices 
    
__all__ = ["MultiUsrpDevice", "MultiUsrpConfiguration", "UsrpConfiguration", "get_device_object", "discover_devices"]
