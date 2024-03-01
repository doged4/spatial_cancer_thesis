# %% Import libraries
import tensorflow as tf
import tensorflow_hub as hub

# %% Define image extracter class

class image_extracter:
    """Wrapper of a keras.sequential model that yields image features
        model_handle: url of tensorflow_hub model to use, str; default effnet_b4
        image_size: image size as height, width, channels as a tuple; default is 380x380x3"""
    def __init__(self, model_handle_url = None, image_size = None, verbose = True):
        # Set default model
        if model_handle_url == None:
            # Efficientnet b4
            self.model_handle_url = "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1"
        else:
            self.model_handle_url = model_handle_url
        
        # Set default image size
        if image_size == None:
            # Found to be closest to spot size
            self.image_size = (380, 380, 3)
        else:
            self.image_size = image_size
        
        self.verbose = verbose # Level of dialogue to user

        if self.verbose:
            print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

        self.model = None # Initialize without model as weights not yet pulled

    def prep_model(self):
        """Pull model weights from tf.hub and setup model"""
        # Initialize model for prediction
        # Model
        #   Input layer of image size
        #   Prebuilt layer
        self.model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape = self.image_size),
                    hub.KerasLayer(self.model_handle_url, trainable = False),
                    ])
        
        self.model.build(self.image_size)
        if self.verbose:
            print("Model ready")

    def extract(self, image):
        """Return extracted image features using a prepped model"""
        if self.model == None:
            raise RuntimeWarning("Prep model before extracting an image")
        return self.model(image)

    

# %%
