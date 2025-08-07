# Try to load all backends and provide 
import sys 

AVAILABLE_BACKENDS = {}

# TODO: Suppress STDOUT
try:
    # Ensure uhd is available
    import uhd # Crashes occur without this
    
    # Ensure device is importable 
    import usrp as usrp

    AVAILABLE_BACKENDS['usrp'] = usrp
except Exception as e: 
    print(f'USRP backend not available: {e}')

try: 
    # Ensure platform is windows 
    if sys.platform != 'win32':
        raise OSError(f'Invalid platfrom {sys.platform}. Ensure you are using Windows')
    import biopac as biopac
    # Ensure mpdev.dll exists 
    if biopac.load_mpdev_dll() == None:
        raise ValueError('mpdev.dll not found')
    
    AVAILABLE_BACKENDS['biopac'] = biopac
except Exception as e:  
    print(f'BIOPAC backend not available: {e}')

__all__ = [
    "AVAILABLE_BACKENDS"
]