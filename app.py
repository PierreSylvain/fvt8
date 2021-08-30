from flask import Flask, redirect, url_for, render_template, request, jsonify
from os import walk
import numpy as np
from tensorflow.python.keras.preprocessing import image
import uuid
import seaborn as sns
import tensorflow as tf
import json
import keras
from azureml.core import Workspace
from azureml.core.model import Model
from keras import backend as K
from azureml.core.authentication import ServicePrincipalAuthentication
import os
from shutil import copyfile


# This is a flask application
app = Flask(__name__)

# Folder for images

filenames = next(walk('/dataset/val/img'), (None, None, []))[2]


# Mask categories
categories = {
    'void': [0, 1, 2, 3, 4, 5, 6],
    'flat': [7, 8, 9, 10],
    'construction': [11, 12, 13, 14, 15, 16],
    'object': [17, 18, 19, 20],
    'nature': [21, 22],
    'sky': [23],
    'human': [24, 25],
    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]
}

categories_color = sns.color_palette("icefire", len(categories))


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3])+K.sum(y_pred, [1, 2, 3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def _dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def dice_loss():
    return _dice_coef_loss


def IoU(y_true, y_pred):
    # mean Intersection over Union
    cats = []
    for cat in categories:
        cats.append(cat)

    IoUs = []
    IoU_details = {}

    for c in range(len(cats)):
        intersection = np.sum((y_pred == c) * (y_true == c))
        y_true_area = np.sum(y_true == c)
        y_pred_area = np.sum(y_pred == c)
        combined_area = y_true_area + y_pred_area
        union_area = combined_area - intersection
        IoU = intersection / combined_area

        index = cats[c]
        IoU_details[index] = IoU

        IoUs.append(IoU)

    return np.mean(IoUs), IoU_details


def set_color_to_mask(mask):
    """Change 8 channels mask to 3 channel RGB
    The colors are set from categories_color array

    Args:
        mask (array): mask as np.array
    """

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # RGB mask (3 channels)
    mask_img = np.zeros((mask.shape[0], mask.shape[1], 3)).astype('float')

    for color_num in range(8):
        # The channel num is the same as color palete index
        mask_color = (mask == color_num)
        mask_img[:, :, 0] += (mask_color *
                              (categories_color[color_num][0]))  # Red
        mask_img[:, :, 1] += (mask_color *
                              (categories_color[color_num][1]))  # Green
        mask_img[:, :, 2] += (mask_color *
                              (categories_color[color_num][2]))  # Blue

    return(mask_img)


def _dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def dice_loss():
    return _dice_coef_loss


def load_data(image_file, height, width):
    """Load mask file from disk

    Args:
        mask_file ([type]): [description]
        height ([type]): [description]
        width ([type]): [description]

    Returns:
        [type]: [description]
    """
    images = []
    masks = []

    mask_file = image_file.replace(
        'leftImg8bit', 'gtFine_labelIds').replace('img', 'mask')
    array_image = image.img_to_array(image.load_img(
        image_file, target_size=(height, width)))/255.
    mask = image.img_to_array(image.load_img(
        mask_file, color_mode="grayscale", target_size=(height, width)))
    mask = np.squeeze(mask)
    array_mask = np.zeros((height, width, 8))

    for i in range(-1, 34):
        if i in categories['void']:
            array_mask[:, :, 0] = np.logical_or(
                array_mask[:, :, 0], (mask == i))
        elif i in categories['flat']:
            array_mask[:, :, 1] = np.logical_or(
                array_mask[:, :, 1], (mask == i))
        elif i in categories['construction']:
            array_mask[:, :, 2] = np.logical_or(
                array_mask[:, :, 2], (mask == i))
        elif i in categories['object']:
            array_mask[:, :, 3] = np.logical_or(
                array_mask[:, :, 3], (mask == i))
        elif i in categories['nature']:
            array_mask[:, :, 4] = np.logical_or(
                array_mask[:, :, 4], (mask == i))
        elif i in categories['sky']:
            array_mask[:, :, 5] = np.logical_or(
                array_mask[:, :, 5], (mask == i))
        elif i in categories['human']:
            array_mask[:, :, 6] = np.logical_or(
                array_mask[:, :, 6], (mask == i))
        elif i in categories['vehicle']:
            array_mask[:, :, 7] = np.logical_or(
                array_mask[:, :, 7], (mask == i))
    images.append(array_image)
    masks.append(array_mask)

    return np.array(images), np.array(masks)



def convert_mask(mask):
    """Convert mask file to RGB file according to color palette and
    save it in static/temp directory with a guid as filename

    Args:
        mask_file (file): File name withn path of mask

    Returns:
        string: RGB mask
    """
    img_mask = set_color_to_mask(mask[0])

    temp_file = "static/temp/" + str(uuid.uuid1()) + ".png"
    tf.keras.preprocessing.image.save_img(
        temp_file,
        img_mask
    )
    return temp_file


def do_prediction(image, mask):
    loaded_model = keras.models.load_model('save_model/save_model/unet', custom_objects={
        'iou_coef': iou_coef,
        'dice_coef': dice_coef,
        '_dice_coef_loss': _dice_coef_loss
    })

    pred = loaded_model.predict(image)
    score = loaded_model.evaluate(image, mask, verbose=0)

    return score, pred


@app.route('/')
def index():
    """ start request"""
    file_count = len(filenames)

    return render_template("index.html", file_count=file_count)


@app.route('/_predict')
def predict():
    """ Ajax request"""

    # Get image id from params
    image_id = request.args.get('image_id', 1, type=int)

    # oad image and mask
    imgs, masks = load_data("/dataset/val/img/" +
                            filenames[image_id], 128, 256)

    # Convert mask
    mask = np.argmax(masks, axis=3)
    converted_mask_file = convert_mask(mask)

    # predict mask and convert it
    score, y_pred = do_prediction(imgs, masks)
    y_predi = np.argmax(y_pred, axis=3)
    pred_mask = convert_mask(y_predi)

    # Get IoU (global IoU mean and IoU by category)
    mean_iou, iou_details = IoU(y_predi, mask)

    copyfile("/dataset/val/img/" + filenames[image_id], "static/temp/" + filenames[image_id])
    
    return jsonify(
        image_orig="static/temp/" + filenames[image_id],
        image_mask=converted_mask_file,
        image_pred=pred_mask,
        accuracy=score[1],
        mean_iou=mean_iou,
        iou=iou_details
    )


# main
if __name__ == "__main__":
    svc_pr_password = os.environ.get("AZUREML_PASSWORD")
        
    svc_pr = ServicePrincipalAuthentication(
        tenant_id="b4d62547-d4ad-4e2d-bbf9-1a33f92310fd",
        service_principal_id="07ea66c7-c56b-471e-b8a3-7a2c37d6c786",
        service_principal_password=svc_pr_password)

    ws = Workspace(
        subscription_id="37490abd-eac8-4845-b5b1-90d998dd6319",
        resource_group="oc-grp",
        workspace_name="fvtp8",
        auth=svc_pr
    )

    model = Model(ws, 'future-vision-transport')
    model.download(target_dir='.', exist_ok=True)
    app.run(debug=True)
