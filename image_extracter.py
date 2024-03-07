# %% Import libraries
import tensorflow as tf
import tensorflow_hub as hub

# %% Define image extracter class

class image_extracter:
    """Wrapper of a keras.sequential model that yields image features
        model_handle: url of tensorflow_hub model to use, str; default effnet_b4
        image_size: image size as height, width, channels as a tuple; default is 380x380x3"""
    def __init__(self, image_size = None, model_handle_url = None, verbose = True):
        # Set default model
        if model_handle_url == None:
            # Efficientnet b4
            self.model_handle_url = "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1"
        else:
            self.model_handle_url = model_handle_url
        
        # Set default image size
        if image_size == None:
            # Found to be closest to spot size for our data
            self.image_size = (380, 380, 3)
        else:
            self.image_size = image_size
        
        self.verbose = verbose # Level of dialogue to user

        if self.verbose:
            print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

        self.model = None # Initialize without model as weights not yet pulled
        self.num_features = None

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
        self.num_features = self.model.output_shape[1]

    def extract_one(self, image, as_np = False):
        """Return extracted image features using a prepped model.
            image: matrix like object of size self.image_size to be coerced to tensor
            as_np: bool, whether to return as np.array"""
        if self.model == None:
            raise RuntimeWarning("Prep model before extracting an image")
        # Check that image will be coercible to our tensor shape
        # This compares (380, 380, 3) with (380, 380, 1, 3) which image containers often have
        assert image.shape[:2] + image.shape[-1:] == self.image_size

        # Convert image array to (self.image_shape) tensor and remove z dim of image_containers
        im_as_tensor =  tf.constant(value=image, shape=self.image_size)
        # Get tensor to (1,) + (self.image_size) right shape to run and make tensor
        im_as_tensor_dataset = tf.expand_dims(im_as_tensor, 0)
        
        # Get results
        features = self.model(im_as_tensor_dataset)

        if as_np:
            return features.numpy()
        else:
            return features
        
    # TODO: test below
    def extract_many(self, images):
        """Return extracted image features using a prepped model"""
        if self.model == None:
            raise RuntimeWarning("Prep model before extracting an image")
        # Make images into tensor to run
        im_tensor = tf.constant(images)

        return self.model(im_tensor)
    

    

# %%
