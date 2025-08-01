import json
import importlib

class Configuration:
    def __init__(self):
        pass 

    def get_disp_freq(self):
        return 10  # Dummy output

    def set_param(self, param, value): 
        current_type = type(getattr(self, param, None))
        if current_type is not None:
            setattr(self, param, current_type(value))
        else:
            setattr(self, param, value)

    def get_param(self, param, default_value=None):
        try:
            value = getattr(self, param)
        except AttributeError:
            value = default_value
        return value
    
    def to_dict(self):
        """Convert object to dictionary for JSON serialization"""
        data = {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_') and not callable(value)}
        
        # Store class information
        data['__class__'] = self.__class__.__name__
        data['__module__'] = self.__class__.__module__
        return data
    
    @classmethod
    def from_dict(cls, data):
        """Create Configuration object from dictionary"""
        # Extract class information
        class_name = data.pop('__class__', None)
        module_name = data.pop('__module__', None)
        
        if class_name and module_name:
            # Dynamically import and instantiate the correct class
            try:
                module = importlib.import_module(module_name)
                target_class = getattr(module, class_name)
                
                # Create instance with empty constructor
                config = target_class.__new__(target_class)
                Configuration.__init__(config)  # Initialize base class
                
                # Set all attributes
                for key, value in data.items():
                    setattr(config, key, value)
                
                return config
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not instantiate {class_name}, falling back to base Configuration")
                # Fall back to base Configuration
                config = cls()
                for key, value in data.items():
                    setattr(config, key, value)
                return config
        else:
            # No class info, use the calling class
            config = cls()
            for key, value in data.items():
                setattr(config, key, value)
            return config
    
    def to_json(self):
        """Serialize to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str):
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))