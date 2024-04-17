# %% Import libraries
import tensorflow as tf
import tensorflow_hub as hub
from anndata import AnnData
from pandas import DataFrame

# %% Define image extracter class

class image_extracter:
    """Wrapper of a keras.sequential model that yields image features
        model_handle: url of tensorflow_hub model to use, str; default effnet_b4
        image_size: image size as height, width, channels as a tuple; default is 380x380x3"""
    def __init__(self, image_size = None, model_handle_url = None, verbose = True):
        # Set default model
        if model_handle_url == None:
            # Efficientnet b4
            self.model_handle_url = "https://www.kaggle.com/models/tensorflow/efficientnet/TensorFlow2/b4-feature-vector/1"
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
        self.image_set = None
        self.image_set_names = None
        self._filenames = None

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
        
    # TODO: function test below
    def extract_many(self, images):
        """Return extracted image features using a prepped model"""
        if self.model == None:
            raise RuntimeWarning("Prep model before extracting an image")
        # Make images into tensor to run
        im_tensor = tf.constant(images)

        return self.model(im_tensor)
    

    def image_set_from_path(self, path, in_place = True, name_append = ""):
        """From path to folder of images generate tensorflow image dataset"""
        if type(path) == str:
            path_as_glob = path
            if path[-1] != '*':
                if path[-1] == '/':
                    path_as_glob += '*'
                else:
                    path_as_glob += '/*'
            self._filenames = tf.data.Dataset.list_files(path_as_glob, shuffle=False)
        elif type(path) == list:
            self._filenames = tf.data.Dataset.list_files(path, shuffle=False)
        
        if type(name_append) == list:
            assert len(name_append) == len(self._filenames)
            dict_result =  {
                    'images': self._filenames.map(image_extracter._read_image),
                    'names' : [image_extracter._read_spot_name(x) + y for x,y in zip(self._filenames, name_append)]
                }
        else:
            dict_result =  {
                'images': self._filenames.map(image_extracter._read_image),
                'names' : [image_extracter._read_spot_name(x) + name_append for x in self._filenames]
            }

        if in_place:
            self.image_set = dict_result
        else:
            return dict_result

    @staticmethod
    def _read_image (path):
        """Convert binary image path to image data as tensor"""
        image_tensor =  tf.image.decode_image(tf.io.read_file(path))
        return tf.expand_dims(image_tensor, axis=0)
    
    @staticmethod
    def _read_spot_name (path):
        """Convert binary image path to spot_id followed by slide id"""
        pathname = path.numpy().decode('ascii')
        filename = pathname.split("\\")[-1]
        split_file_name = filename.split("_")
        barcode = split_file_name[2]
        slide_id = split_file_name[0][-2:] + split_file_name[1]

        return barcode + '_' + slide_id

    def extract_from_imageset(self, image_set = None, use_multiprocessing = True, **kwargs):
        """Extract image features from image dataset.
            Args:
                image_set : tf.data.Dataset of images, can be none and self.image_set[0] will be used
                use_multiprocessing : boolean to pass to predict function
            Return:
                image_result as np.array
        """
        if image_set == None:
            image_set = self.image_set['images']
        
        image_result =  self.model.predict(x = image_set, use_multiprocessing = use_multiprocessing, **kwargs)

        return image_result
    



    def extract_from_imageset_adata(self, image_set = None, names = None, 
                                    use_multiprocessing = True, name_append = "", **kwargs):
        array_result = self.extract_from_imageset(image_set, use_multiprocessing, **kwargs)
        if names == None:
            if name_append == "":
                image_names = self.image_set['names']
            else:
                image_names = [x + name_append for x in self.image_set['names']]
        else: 
            image_names = names
        return image_extracter._imageset_to_adata(image_result = array_result,
                                                  image_names = image_names)

    @staticmethod
    def _imageset_to_adata(image_result, image_names):

        return AnnData(
            X = image_result,
            obs = DataFrame({'spot_ids' : image_names}, index = image_names),
            var = DataFrame({'feature' :[f"feature_{i}" for i in range(image_result.shape[1])]},
                            index = [f"feature_{i}" for i in range(image_result.shape[1])])
        )

# %%