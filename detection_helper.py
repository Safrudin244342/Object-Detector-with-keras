import cv2

def sliding_window(image, step, ws):
    # slide window across image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            result = (x, y, image[y:y + ws[1], x:x + ws[0]])
            yield result


def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    # yield the original image
    yield image

    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in pyramid
        w = int(image.shape[0] / scale)
        h = int(image.shape[1] / scale)
        image = cv2.resize(image, (w, h))

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image