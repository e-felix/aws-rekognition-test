import boto3
import io
from PIL import Image, ImageDraw, ImageFilter

client = boto3.client('rekognition')

with open('image.jpg', 'rb') as imageFile:
    faces = {}
    face = {}

    img = imageFile.read()

    response = client.detect_faces(
        Image={
            'Bytes': img
        }
    )

    image = Image.open(io.BytesIO(img))
    imgWidth, imgHeight = image.size
    draw = ImageDraw.Draw(image)

    for index, faceDetail in enumerate(response['FaceDetails']):
        box = faceDetail['BoundingBox']
        left = int(imgWidth * box['Left'])
        top = int(imgHeight * box['Top'])
        width = int(imgWidth * box['Width'])
        height = int(imgHeight * box['Height'])

        print(f"confidence: {faceDetail['Confidence']}")
        print(f"Left: {left}")
        print(f"Top: {top}")
        print(f"Face Width: {width}")
        print(f"Face Height: {height}")

        points = (
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
            (left, top)
        )

        faces[index] = {
            'leftBottom': left,
            'leftTop': top,
            'rightBottom': left + width,
            'rightTop': top + height,
            'cropImg': image.crop((left, top, left + width, top + height))
        }

    for face in faces.values():
        leftBottom = face.get('leftBottom')
        leftTop = face.get('leftTop')
        rightBottom = face.get('rightBottom')
        rightTop = face.get('rightTop')
        crop_img = face.get('cropImg')

        blurred = crop_img.filter(ImageFilter.GaussianBlur(8))
        crop_img.paste(blurred)
        image.paste(blurred, (leftBottom, leftTop, rightBottom, rightTop))

    image.show()
