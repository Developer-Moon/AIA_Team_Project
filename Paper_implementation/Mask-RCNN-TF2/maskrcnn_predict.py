import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os


# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath="D:\_AIA_Team_Project_Data\paper\maskrcnn\mask_rcnn_coco.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
# image = cv2.imread("C:\study\Mask-RCNN-TF2-master/test.jpg")
image = cv2.imread("D:\_AIA_Team_Project_Data\Image_Captioning\_test/0000.jpg")

# image folder
# image = "C:\study\Mask-RCNN-TF2-master\images"

# image_list = []

# # loop over the image folder
# for filename in os.listdir(image):
#     # load the input image, convert it from BGR to RGB channel
#     image = cv2.imread(os.path.join(image, filename))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_list.append(image)
    
#     # perform a forward pass of the network to obtain the results
#     results = model.detect([image], verbose=1)
    
#     # extract the results for the first image
#     r = results[0]
    
#     # visualize the results of the detection
#     mrcnn.visualize.display_instances(image=image,
#                                                 boxes=r['rois'],
#                                                 masks=r['masks'],
#                                                 class_ids=r['class_ids'],
#                                                 class_names=CLASS_NAMES,
#                                                 scores=r['scores'])
    



# output folder

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image])

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])
