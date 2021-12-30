import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class TFImageDataGenerator:
    
    def __init__(
        self,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        dtype=None,
    ):
        self.featurewise_center=featurewise_center
        self.samplewise_center=samplewise_center
        self.featurewise_std_normalization=featurewise_std_normalization
        self.samplewise_std_normalization=samplewise_std_normalization
        self.zca_whitening=zca_whitening
        self.zca_epsilon=zca_epsilon
        self.rotation_range=rotation_range
        self.width_shift_range=width_shift_range*0.8
        self.height_shift_range=height_shift_range*0.8
        if isinstance(brightness_range, list) and len(brightness_range) == 2:
            self.brightness_range=[brightness_range[0]*0.6,brightness_range[1]*0.6]
        else:
            self.brightness_range=brightness_range
        self.shear_range=round(shear_range*0.8)
        self.zoom_range=zoom_range*0.8
        self.channel_shift_range=channel_shift_range
        self.fill_mode=fill_mode
        self.cval=cval
        self.horizontal_flip=horizontal_flip
        self.vertical_flip=vertical_flip
        self.rescale=rescale
        self.preprocessing_function=preprocessing_function
        self.data_format=data_format
        self.validation_split=validation_split
        self.dtype=dtype
        
        self.interpolation = 'lanczos'
        self.target_size=(224, 224)
    
    def image_tensor_from_path(self,path):
        image_tensor = tf.io.read_file(path)
        image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
        image_tensor = tf.image.resize(image_tensor, self.target_size, method='area')
        return image_tensor
    
    def random_transform(self,image,seed=None):
        image = tf.cast(image, tf.float32)
        image_width, image_height, image_colors = image.shape
        
        fill_mode = self.fill_mode
        interpolation=self.interpolation
        if not interpolation == 'nearest':
            interpolation = 'bilinear'

        # width shift
        if not self.width_shift_range == 0:
            factor = tf.random.uniform(shape=[], minval=-self.width_shift_range, maxval=self.width_shift_range, dtype=tf.float32)
            image = tfa.image.translate(image, translations=[factor*image_width,0],interpolation=interpolation, fill_mode=fill_mode)
            
        # height shift
        if not self.height_shift_range == 0:
            factor = tf.random.uniform(shape=[], minval=-self.height_shift_range, maxval=self.height_shift_range, dtype=tf.float32)
            image = tfa.image.translate(image, translations=[0,factor*image_height],interpolation=interpolation, fill_mode=fill_mode)       
        
        # zoom
        if not self.zoom_range == 0:
            crop_size = (image_width, image_height)      
            scales = list(np.arange(1-self.zoom_range, 1+self.zoom_range, 0.01))
            boxes = np.zeros((len(scales), 4))
            for i, scale in enumerate(scales):               
                x1 = y1 = 0.5 - (0.5 * scale)  
                x2 = y2 = 0.5 + (0.5 * scale)  
                boxes[i] = [x1, y1, x2, y2]
                
            def random_crop(img):   # Create different crops for an image
                crops = tf.image.crop_and_resize( [img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=crop_size )
                return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)] 
            image = random_crop(image)

        # rotation
        if not self.rotation_range == 0:
            factor = tf.random.uniform(shape=[], minval=-self.rotation_range, maxval=self.rotation_range, dtype=tf.float32)
            image = tfa.image.rotate(image, factor * np.pi / 180, interpolation=interpolation, fill_mode=fill_mode)

        # shear 
        if not self.shear_range == 0:
            factor = tf.random.uniform(shape=[], minval=-self.shear_range, maxval=self.shear_range, dtype=tf.int32)
            #image = tfa.image.shear_y( image, factor/180, 0 )
            rot = (factor/45)
            image = tf.image.rot90(image)
            image = tfa.image.transform(image, [1.0, rot, -rot*image_height/2, 0.0, 1.0, 1.0, 0.0, 0.0])
            image = tf.image.rot90(image,k=-1)


        # horizontal_flip
        if self.horizontal_flip:
            image = tf.image.random_flip_left_right(image)
        
        # vertical_flip
        if self.vertical_flip:
            image = tf.image.random_flip_up_down(image)

        # random brightness
        if isinstance(self.brightness_range, list) and len(self.brightness_range) == 2:
            rn = self.brightness_range[1] - self.brightness_range[0]
            image = tf.image.random_brightness(image, rn*255/2)

        # preprocessing
        if callable(self.preprocessing_function):
            image = self.preprocessing_function(image)
            
        return image
    
    def flow(
        self,
        x,
        y=None,
        batch_size=32,
        shuffle=True,
        sample_weight=None,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
    ):
        AUTOTUNE = tf.data.AUTOTUNE
        
        out = tf.data.Dataset.from_tensor_slices(( x, y )).map( lambda image, label: (self.random_transform(image), label), num_parallel_calls=AUTOTUNE )
            
        if shuffle:
            out = out.repeat().shuffle( len(x) )
                
        out = out.batch(batch_size)
        
        if not shuffle:
            out = out.cache()
        
        return out.prefetch(buffer_size=AUTOTUNE)
    
    def flow_from_dataframe(
        self,
        dataframe,
        directory=None,
        x_col="filename",
        y_col="class",
        weight_col=None,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
        interpolation="nearest",
        validate_filenames=True,
        **kwargs
    ):
        self.target_size = target_size
        self.interpolation = interpolation
        AUTOTUNE = tf.data.AUTOTUNE
        
        lb = MultiLabelBinarizer()
        labels = lb.fit_transform(dataframe[y_col])
        
        out = tf.data.Dataset.from_tensor_slices(( dataframe[x_col], labels )).map( lambda path, label: (self.random_transform(self.image_tensor_from_path(path)), label), num_parallel_calls=AUTOTUNE )
            
        if shuffle:
            out = out.repeat().shuffle( len(dataframe[x_col]) )
                
        out = out.batch(batch_size)
        
        if not shuffle:
            out = out.cache()
        
        return out.prefetch(buffer_size=AUTOTUNE)