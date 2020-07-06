## ## This file was created by Geeta Garg, Indiana University (email: gegarg@iu.edu) to convert png transcripts (downloaded in pdf format from Federal Reserve's website) to readable text format for a summer project
## This is the only file that needs to be used to convert png to text.
## This file can run a loop over png files saved in directory_in and saves them in directory_text with text file using the base name of the pdf file.
## This file converts noisy scanned pdf images into clear text using Python and Google Tesseract OCR (The input should be png files and output will be text)
## This file is created based on different sources listed below:
## Source 1.: https://blog.anirudhmergu.com/code/ocr-python-tesseract-ocr/
## Source 2: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
## Source 3: https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
## Source 4: Image preprocessing options: https://auth0.com/blog/image-processing-in-python-with-pillow/

######################################################################################################

import cv2
import pytesseract
from PIL import Image
import os
import pickle
import pandas as pd
import re
#import datetime
import numpy as np
import warnings
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


def main():
    filename_all = []
    text_corpus = []
    file_array = []
    all_lines = []
  
    directory_in = '/Users/geetagarg/fed_pdf_final/new_trial'
    directory_text = '/Users/geetagarg/fed_text_compare'
    # Get File Name from Command Line

    for file in os.listdir(directory_in):
        #       print("file", file)

        if file.startswith('.') and os.path.isfile(os.path.join(directory_in, file)):
            print("error_file", file)
            continue
        file_array.append(file)
    # print("file_array", file_array)

    file_array = [os.path.join(directory_in, file) for file in file_array]
    print("file_array_1", file_array)


    for files_1 in file_array:
        print("files_1", files_1)
        file_name = files_1.split("/")[-1]
        print("file_name", file_name)
        #Locate the png file to be converted to text
        path = files_1
        # load the image
        image = cv2.imread(path)
        # Rescale the image, if needed.
        #image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)  #### new
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(gray, kernel, iterations=1) # removes small white noises
        img = cv2.erode(img, kernel, iterations=1)  # once noise is removed in the previous step, dilate the image for better clarity
        # Apply Gaussian blur to smooth out the edges/noise
        img = cv2.GaussianBlur(img, (5, 5), 0)   ## This is newly added and can be removed. This reads the words a little better but also gets some words wrong which are correct without including it (choose alternate filters if this does not work)
        gray1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]   ## Using 127 instead of 0 brings more clarity to the text
        
        # store grayscale image as a temp file to apply OCR
        filename = "{}.png".format("temp")
        cv2.imwrite(filename, gray1)

    # load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file
        text = pytesseract.image_to_string(Image.open(filename))
        print("lengthoftext", len(text))
        print("OCR Text is " + text)

        ocrpdf2text = open(directory_text + '/' + file_name.split(".")[0] +'.txt', 'w')
        #  for text_para in text:
        ocrpdf2text.write(text)
    
    
    #fname = directory_text + '/' + file_name.split(".")[0] +'.txt'
    #   print("fname", fname)
    #   with open(fname, "r") as f:
    #        head = list(islice(f, 6))
    #        print("head", head)
    #        with open('output_firstline.txt', "w") as f2:
    #            for item in head:
    #                f2.write(item)

        text_corpus.append(text)
        filename_all.append(file_name)
        print("filename_all", filename_all)
#print("length_textcorpus", len(text_corpus))

#Data = {'Filename': file_name, 'Pub_Date': date, 'Content': text_corpus}
        Data = {'Filename': filename_all, 'Content': text_corpus}
        dataframe = pd.DataFrame(data=Data)
        print(dataframe)

        with open('newshist_txt_save_more.pkl', 'wb') as picklefile:
            pickle.dump(dataframe, picklefile)


######################################################################################################
######################################################################################################
try:
    main()
except Exception as e:
    print(e.args)
    print(e.__cause__)

## Use as: python png_to_text.py
    





