from PIL import Image
import os
import glob
from SOURCE.yolo_files import detect
from helper_fns import gan_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

YOLO_RESULT = 'results/yolov5/'
YOLO_OP = 'crops/DLSignature/'

def signature_detection(image_dir, selection=None):
    """
    Performs signature detection on images in the specified directory and returns the results folder path.
    
    Parameters:
        image_dir (str): The directory containing images for detection.
        selection (str, optional): Filename of a specific image to display the detection result.

    Returns:
        str: Path to the folder containing cropped detection results.
    """
    
    # Run YOLO detection on all images in the provided directory.
    detect.detect(image_dir)
    
    # Get the path of the latest detection results folder.
    latest_detection = max(glob.glob(os.path.join(YOLO_RESULT, '*/')), key=os.path.getmtime)
    
    # Resize detected signature images and add padding as required by the GAN model.
    gan_utils.resize_images(os.path.join(latest_detection, YOLO_OP))
    
    # Optionally, display a specific detection if the selection is provided.
    if selection:
        selection_detection = os.path.join(latest_detection, YOLO_OP, f'{selection}.jpg')
        if os.path.exists(selection_detection):
            # Display image using Matplotlib
            img = mpimg.imread(selection_detection)
            plt.imshow(img)
            plt.axis('off')  # Hide axes
            plt.show()
        else:
            print(f"Warning: The selected file '{selection}.jpg' was not found in the detection results.")

    
    # Return the path to the folder containing the detected and cropped signature images.
    return os.path.join(latest_detection, YOLO_OP)

image_directory = 'C:/Users/lenovo/Desktop/Stage_SG/Signature detecion/App/media/documents'
result_folder = signature_detection(image_directory)
print(f"Detection results saved in: {result_folder}")
