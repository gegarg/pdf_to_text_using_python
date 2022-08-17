## ## This file was created by Geeta Garg, Indiana University (email: gegarg@iu.edu) to convert pdf transcripts downloaded from 
## Federal Reserve's website to png format for a summer project
## This file converts pdf to png files and saves them in fed_files_pdf_jpg folder
## When the files are converted from pdf to png they are split into separate png pages which are then needed to be combined together 
## to form one png file. The first image becomes black when combined with the second page/image of the same article 
## (manually keep checking articles if an article is damaged)

## https://blog.softhints.com/python-extract-text-from-image-or-pdf/
## http://xiaofeima1990.github.io/2016/12/19/extract-text-from-sanned-pdf/#python-OCR-stript
## https://gist.github.com/jweir/7ce4eb2ac5f7c9b8fe015fe4278a6800
####################################################################################################################################
## Importing required libraries
####################################################################################################################################

from wand.image import Image
import warnings
from PIL import Image as PI
import pyocr
import pyocr.builders
import io
import os
import pdf2image
from pdf2image import convert_from_path
from wand.color import Color
#from PIL import Image
PI.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', PI.DecompressionBombWarning)
#import cv2
#import numpy
#import glob


####################################################################################################################################
## get the handle of the OCR library (tesseract)
####################################################################################################################################
tool = pyocr.get_available_tools()[0]
lang = tool.get_available_languages()[0]  # you need to check what the language is in the list, in my computer it is eng for [0]


####################################################################################################################################
## setup two lists to store the images and final_text
####################################################################################################################################

req_image = []
final_text = []


####################################################################################################################################
## open the PDF file using wand and convert it to jpeg
####################################################################################################################################

directory = '/Users/geetagarg/fed_pdf_files/new_trial'
directory_out = '/Users/geetagarg/fed_pdf_files_jpg'
directory_final = '/Users/geetagarg/fed_pdf_final/new_trial'

os.chdir(directory)

file_array = []
for file in os.listdir(directory):
    print("file", file)
    if file.startswith('.') and os.path.isfile(os.path.join(directory, file)):
        print("error_file", file)
        continue
    file_array.append(file)
    print("file_array", file_array)

#file_array = [os.path.join(directory, file) for file in file_array]
#print("file_array_1", file_array)

jpeg_images = []

for files_1 in file_array:
    print("files_1", files_1)
    image_pdf = Image(filename=files_1, resolution=400) # Image is in pdf format
    image_jpeg = image_pdf.convert('png')
    image_jpeg.background_color = Color('white')
    image_jpeg.alpha_channel = 'remove'
    image_jpeg.compression_quality = 90
    file_name = files_1.split(".")[0]
    print("file_name", file_name)
    #save_png = image_jpeg.save(filename = file_name+'.png')  # saving the files as png works instead of jpg
    page_no = len(image_jpeg.sequence) # get the number of pages
    print("number", page_no)
    
    if page_no == 1:
        image1 = Image(image_jpeg).save(filename= directory_final + '/'  + file_name + '.png')  # For images with only one page

    else:
        for i in range(0, len(image_jpeg.sequence)): # for images with more than one page, save each page separately and join them later
            image1 = Image(image_jpeg.sequence[i]).save(filename= directory_out + '/'  + file_name + '-' + str(i) + '.png')  # save the image using pdf file base name and the page number as the identifier
                
        if page_no > 1:
            save_image = []
            save_width = []
            save_height = []
            range1 = range(page_no)
            print("range", range(page_no))
            print("range1", range1[-1])
            for i in range(page_no):
        
        #image1 = image_jpeg.save(filename=file_name +'.png')
        #else:
        # image_to_append = image_jpeg.sequence[i]
        #save_image.append(image_to_append)
        # image1 = Image(image_jpeg.sequence[i]).save(filename= directory_out + file_name +str(i)+'.png')
                print(directory_out + '/' + file_name + '-' + str(i)+'.png')
                image_open = PI.open(directory_out + '/' + file_name + '-' + str(i)+'.png')
                    #print("image_open", str(image_open))
                save_image.append(image_open)
                #print("length", len(save_image))

                (width, height) = image_open.size   # Get the width and height of each image/page in image
                #print("width", width)
                save_width.append(width)
                save_height.append(height)
                # print("save_height", save_height[i])
                #print("len_save_height", len(save_height))
            total_width = 0
            total_height = 0
            index = 0
            while index < len(save_height):
                # print("index", index)
                #total_height = total_height + save_height[index]
                total_width = total_width + save_width[index]
                #print("total_width", total_width)
                index = index + 1
                # print("index1", index)
                # result_width = max(save_width[0], save_width[1])
                result_width = total_width
                # print("result_width", result_width)
                result_height = max(save_height[0], save_height[1])

            result = PI.new('RGB', (result_width, result_height), "white")

            result.paste(im=save_image[0], box=(0, 0))
            result.paste(im=save_image[1], box=(save_width[0], 0))
            result.save(directory_final + '/' + file_name +'.png')  # save




#   with open(out.jpg, "wb") as jpgfile:
#            jpgfile.write(image_jpeg)


####################################################################################################################################
## wand has converted all the separate pages in the PDF into separate image blobs. We can loop over them and append them as a blob into the req_image list.
####################################################################################################################################


#  for img in image_jpeg.sequence:
#        img_page = Image(image=img)
#        req_image.append(img_page.make_blob('png'))



