import boto3
import io
import re
from PIL import Image, ImageDraw, ImageFilter

client = boto3.client('rekognition')

def detectFaces(inputImage, outputImage):
    faces = {}

    imgFaces = client.detect_faces(
        Image={
            'Bytes': img
        }
    )

    if len(imgFaces["FaceDetails"]) > 0:
        for index, faceDetail in enumerate(imgFaces['FaceDetails']):
            box = faceDetail['BoundingBox']
            left = int(imgWidth * box['Left'])
            top = int(imgHeight * box['Top'])
            width = int(imgWidth * box['Width'] * 1.2)
            height = int(imgHeight * box['Height'] * 1.2)

            faces[index] = {
                'leftBottom': left,
                'leftTop': top,
                'rightBottom': left + width,
                'rightTop': top + height,
                'cropImg': image.crop((left, top, left + width, top + height))
            }

    return faces

def detectLicensePlates(inputImage, outputImage):
    licences = {}

    licensePlates = client.detect_text(
        Image={
            'Bytes': img
        }
    )

    if len(licensePlates['TextDetections']) > 0:
        for index, licensePlate in enumerate(licensePlates['TextDetections']):
            match = re.fullmatch("^[aA-zZ]{2}-[0-9]{3}-[aA-zZ]{2}$", licensePlate["DetectedText"])
            if match:
                box = licensePlate['Geometry']['BoundingBox']
                left = int(imgWidth * box['Left'])
                top = int(imgHeight * box['Top'])
                width = int(imgWidth * box['Width'] * 1.2)
                height = int(imgHeight * box['Height'] * 1.2)

                licences[index] = {
                    'leftBottom': left,
                    'leftTop': top,
                    'rightBottom': left + width,
                    'rightTop': top + height,
                    'cropImg': image.crop((left, top, left + width, top + height))
                }

    return licences

with open('voiture_woman.jpg', 'rb') as imageFile:
    img = imageFile.read()

    image = Image.open(io.BytesIO(img))
    imgWidth, imgHeight = image.size
    draw = ImageDraw.Draw(image)

    faces = detectFaces(img, image)
    licences = detectLicensePlates(img, image)

    for face in faces.values():
        leftBottom = face.get('leftBottom')
        leftTop = face.get('leftTop')
        rightBottom = face.get('rightBottom')
        rightTop = face.get('rightTop')
        crop_img = face.get('cropImg')

        blurred = crop_img.filter(ImageFilter.GaussianBlur(8))
        crop_img.paste(blurred)
        image.paste(blurred, (leftBottom, leftTop, rightBottom, rightTop))

    for license in licences.values():
        leftBottom = license.get('leftBottom')
        leftTop = license.get('leftTop')
        rightBottom = license.get('rightBottom')
        rightTop = license.get('rightTop')
        crop_img = license.get('cropImg')

        blurred = crop_img.filter(ImageFilter.GaussianBlur(8))
        crop_img.paste(blurred)
        image.paste(blurred, (leftBottom, leftTop, rightBottom, rightTop))

    image.show()
