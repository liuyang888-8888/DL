from onnx_nonms import predict
import matplotlib.pyplot as plt
import numpy as np
import cv2



model = predict()

if __name__ == '__main__':

    # result1,box=model.detect_image(r'C:\Users\nxxia\Desktop\project2\Desktop\B2.jpg')
    model = predict()
    result=model.detect_image(r'C:\Users\nxxia\Desktop\project2\Desktop\B2.jpg')


    print(result)