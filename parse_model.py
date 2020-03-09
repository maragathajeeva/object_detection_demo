from django.shortcuts import render
from rest_framework import views
from rest_framework.response import Response
from json import dumps, loads, JSONEncoder, JSONDecoder
import requests
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .parse_serializers import BaseSearchSerializer
# Python Code Modules
import os
from PIL import Image, ImageEnhance
import json
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy
from django.http import JsonResponse

import PIL
import numpy as np
from PIL import Image, ImageDraw, ExifTags, ImageColor
from PIL import Image
import io
import random

import cv2
import logging
import numpy as np
import tensorflow as tf
import pytesseract
import imutils
import shutil
import glob
import string
import re
import sys
import pytesseract
import sqlalchemy
from flashtext import KeywordProcessor

import pymysql
import pymysql.cursors
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool
sys.path.append("..")
pymysql.install_as_MySQLdb()
from myproject.object_detection.utils import label_map_util
from myproject.object_detection.utils import visualization_utils as vis_util
import tesserocr
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\mohanraj\AppData\Local\Tesseract-OCR\\tesseract.exe"
output_path = r"C:\Users\mohanraj\PycharmProjects\Django-app\venv\turbo\django_proj\django_proj\myproject\Output"
input_path = r'C:\Users\mohanraj\PycharmProjects\Django-app\venv\turbo\django_proj\django_proj\myproject\Extract'
dirs = ['COMPLIANCE_REPORT', 'CORRESPONDENCE', 'DEMOGRAPHICS', 'PATIENT_INSURANCE_PAYMENT', 'RX', 'SETUP_TICKET',
        'SLEEP_STUDY_REPORT', 'INSURANCE_VERIFICATION', 'MASK_GUARANTEE_FORM', 'SHIPPET_ITEMS', 'others']
NUM_CLASSES = 90
model1_PATH_TO_CKPT = r'models/model_1' + '/frozen_inference_graph.pb'
label_path1 = r'models/model_1'
PATH_TO_LABELS1 = os.path.join(label_path1, 'label_map.pbtxt')
label_map1 = label_map_util.load_labelmap(PATH_TO_LABELS1)
categories1 = label_map_util.convert_label_map_to_categories(label_map1, max_num_classes=NUM_CLASSES,
                                                             use_display_name=True)

model2_PATH_TO_CKPT = r'models/model_2' + '/frozen_inference_graph.pb'
label_path2 = r'models/model_2'
PATH_TO_LABELS2 = os.path.join(label_path2, 'label_map.pbtxt')

label_map2 = label_map_util.load_labelmap(PATH_TO_LABELS2)
categories2 = label_map_util.convert_label_map_to_categories(label_map2, max_num_classes=NUM_CLASSES,
                                                             use_display_name=True)

detection_graph1 = tf.Graph()
with detection_graph1.as_default():
        od_graph_def1 = tf.GraphDef()
        with tf.gfile.GFile(model1_PATH_TO_CKPT, 'rb') as fid1:
            serialized_graph1 = fid1.read()
            od_graph_def1.ParseFromString(serialized_graph1)
            tf.import_graph_def(od_graph_def1, name='')
detection_graph2 = tf.Graph()
with detection_graph2.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model2_PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

pytesseract.pytesseract.OMP_THREAD_LIMIT = 3
Image.MAX_IMAGE_PIXELS = None

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='turbosmart',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
class ParseModel:

    def __init__(self):
        print("in init")

    def get_folder_path(self, path):
        file_list = []
        for file in os.listdir(path):
            file_list.append(str(path) + '\\' + str(file))
        object_dict = {}
        object_dict['file_list'] = {}

        object_dict['file_list'] = file_list
        object_dict['total'] = len(file_list)
        print('we', object_dict)
        return object_dict['file_list']

    def file_logging(self, method, datas, directory, input):
        main = logging.getLogger('main')
        main.setLevel(logging.DEBUG)
        handler = logging.FileHandler(output_path + '/' + 'Process_Log.log')

        format = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(format)
        main.addHandler(handler)
        # logging.basicConfig(filename=output_path + '/' + 'Process_Log.log',
        #                     format='%(asctime)s %(message)s',
        #                     filemode='w')

        # Setting the threshold of logger to DEBUG

        if (method == 'objectdetection'):
            main.debug("ImageName:" + os.path.basename(
                input_path + "/" + input) + "  Output:{" + str(datas[0]) + ":" + str(
                datas[1]) + "}" + "  Move_status: " + directory)

        elif (method == 'keywordmatching'):
            main.debug("ImageName:" + input + "  keywords_match:{" + str(datas) + "}" + "  Move_status: " + directory)
        main.removeHandler(handler)
        return main

    def keyword_processing(self, image_to_text):
       
        punctuations = '''|!()-[]{};:=-—'"\,<>./?@$%^&*_~'''

        for x in image_to_text.lower():
            if x in punctuations:
                image_to_text = image_to_text.replace(x, "")
        image_tot_text = ''.join(image_to_text.splitlines())
        b = sent_tokenize(image_to_text)
        image_to_texts = []
        for a in b:
            d = TextBlob(a)
            image_to_texts.append(d.correct())
        for text in image_to_texts:
            image_to_text = str(text)



        with connection.cursor() as cursor:
                sql = '''SELECT  (select categoryName from mstprocesscategory where categoryId=keyword.categoryId LIMIT 1) Category_name, 
                        keyword,categoryId FROM `trnprocesscategorykeyword` keyword WHERE MATCH(keyword)
                        AGAINST(%s IN NATURAL LANGUAGE MODE)'''

                cursor.execute(sql,image_to_text)
                result_set = cursor.fetchall()
                print(result_set)

        dict_result = {}
        results = []
        for resul in result_set:
            for column, value in resul.items():
                if (column == 'keyword'):
                    keyword_processor = KeywordProcessor(case_sensitive=False)
                    keyword_processor.add_keyword(str(value))
                    keywords_found = keyword_processor.extract_keywords(image_to_text)

                    if (keywords_found):

                        if ('None' not in keywords_found):
                             results.append((result_set[result_set.index(resul)]['Category_name'], keywords_found))
                    #keyword_processor.remove_keyword(str(value))

        column_name = []
        keywords_match = []
        if (results != []):
            for result in results:
                (a, b) = result
                column_name.append(a)
                keywords_match.append(b)
        return (column_name, keywords_match,results)

    def keyword_find(self, fname, total, index):

        # image= cv2.imread(r'C:\Users\mohanraj\Desktop\der\others\2-04.png')

        # image=cv2.imread(r'C:\Users\MURUGAS\Desktop\der\others\1-06.png')
        # -- opencv to PIL
        # image = Image.open(fname).convert('RGB')
        # open_cv_image = np.array(image)
        # # Convert RGB to BGR
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        #image2 = cv2.resize(open_cv_image, None, fx=20, fy=20, interpolation=cv2.INTER_LINEAR_EXACT)

        image = cv2.imread(fname, 0)
        image_size = cv2.resize(image, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
        thresh = cv2.threshold(image_size, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        result = 255 - thresh

        image_to_text = pytesseract.image_to_string(result)
        if(image_to_text==''):
            thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            result = 255 - thresh

            image_to_text = pytesseract.image_to_string(result)
        # image = cv2.imread(fname)
        #
        image_np = cv2.imread(fname)
        #
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #
        # kernel = np.ones((1, 1), np.uint8)
        # image = cv2.dilate(image, kernel, iterations=1)
        # images = cv2.erode(image, kernel, iterations=1)
        # image_to_text1 = ''
        # image_to_text2 = ''
        # image_to_text3 = ''
        # image_to_text4 = ''
        # image_to_text5 = ''
        #
        # try:
        #     #pytesseract.pytesseract.tesseract_cmd = r"C:\Users\mohanraj\AppData\Local\Tesseract-OCR\\tesseract.exe"
        #     image = cv2.resize(open_cv_image, None, fx=20, fy=20, interpolation=cv2.INTER_LINEAR)
        #     images = Image.fromarray(
        #         cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     )
        #     image_to_text1 = tesserocr.image_to_text(images)
        #
        # except:
        #
        #     try:
        #         image = cv2.resize(open_cv_image, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
        #         images = Image.fromarray(
        #             cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #         )
        #         image_to_text2 = tesserocr.image_to_text(images)
        #     except:
        #         try:
        #             image = cv2.resize(open_cv_image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
        #             images = Image.fromarray(
        #                 cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #             )
        #             image_to_text3 = tesserocr.image_to_text(images)
        #
        #         except:
        #             try:
        #                 images = Image.fromarray(
        #                     cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        #                 )
        #                 image_to_text4 = tesserocr.image_to_text(images)
        #             except:
        #                 print(str(os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))))
        #                 cv2.imwrite(output_path + '/' + dirs[10] + '/' + str(
        #                     os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))),
        #                             image_np)
        #
        # if ((image_to_text1 == '') and (image_to_text2 == '') and (image_to_text3 == '') and (image_to_text4 == '')):
        #     try:
        #         images = Image.fromarray(
        #             cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        #         )
        #         image_to_text5 = pytesseract.image_to_string(images)
        #     except:
        #         print(str(os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))))
        #         cv2.imwrite(output_path + '/' + dirs[10] + '/' + str(
        #             os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))), image_np)
        #
        # image_to_text = [image_to_text1, image_to_text2, image_to_text3, image_to_text4, image_to_text5]
        # keywords_match = []
        # k = 0
        # while (k <= 4):
        #     if ((len(keywords_match) == 0) or (len(keywords_match) == 1)):
        #
        #
        #         (column_name, keywords_match,results) = self.keyword_processing(image_to_text[k])
        #     k = k + 1
        
        (column_name, keywords_match,results) = self.keyword_processing(image_to_text)
        print('hhh', column_name)
        if (len(keywords_match) >= 1):
            if (len(set(column_name)) > 1):
                if (('SLEEP_STUDY_REPORT' in (set(column_name))) and ('COMPLIANCE_REPORT' in (set(column_name)))):

                    for col in column_name:
                        if (col == 'SLEEP_STUDY_REPORT'):

                            for result in results:
                                (c, d) = result
                                if (col == c):
                                    results.remove(result)
                                    column_name.remove(c)
                                    keywords_match.append(d)
                if (('COMPLIANCE_REPORT' in (set(column_name))) and ('CORRESPONDENCE' in (set(column_name)))):

                    for col in column_name:
                        if (col == 'CORRESPONDENCE'):

                            for result in results:
                                (c, d) = result
                                if (col == c):
                                    results.remove(result)
                                    column_name.remove(c)
                                    keywords_match.append(d)
                        elif (col == 'COMPLIANCE_REPORT'):

                            for result in results:
                                (c, d) = result
                                if (col == c):
                                    results.remove(result)
                                    column_name.remove(c)
                                    keywords_match.append(d)
                if (('RX' in (set(column_name))) and ('CORRESPONDENCE' in (set(column_name)))):

                    for col in column_name:
                        if (col == 'CORRESPONDENCE'):

                            for result in results:
                                (c, d) = result
                                if (col == c):
                                    results.remove(result)
                                    column_name.remove(c)
                                    keywords_match.append(d)
                        if (col == 'RX'):

                            for result in results:
                                (c, d) = result
                                if (col == c):
                                    results.remove(result)
                                    column_name.remove(c)
                                    keywords_match.append(d)
                if (('DEMOGRAPHICS' in (set(column_name))) and ('CORRESPONDENCE' in (set(column_name)))):

                    for col in column_name:
                        if (col == 'CORRESPONDENCE'):

                            for result in results:
                                (c, d) = result
                                if (col == c):
                                    results.remove(result)
                                    column_name.remove(c)
                                    keywords_match.append(d)
                        if (col == 'DEMOGRAPHICS'):

                            for result in results:
                                (c, d) = result
                                if (col == c):
                                    results.remove(result)
                                    column_name.remove(c)
                                    keywords_match.append(d)
                if (('RX' in (set(column_name))) and ('INSURANCE_VERIFICATION' in (set(column_name)))):

                    for col in column_name:
                        if (col == 'INSURANCE_VERIFICATION'):

                            for result in results:
                                (c, d) = result
                                if (col == c):
                                    results.remove(result)
                                    column_name.remove(c)
                                    keywords_match.append(d)
                if (('RX' in (set(column_name))) and ('PATIENT_INSURANCE_PAYMENT' in (set(column_name)))):

                    for col in column_name:
                        if (col == 'PATIENT_INSURANCE_PAYMENT'):

                            for result in results:
                                (c, d) = result
                                if (col == c):
                                    results.remove(result)
                                    column_name.remove(c)
                                    keywords_match.append(d)


                if (max(set(column_name), key=column_name.count)):
                    datas = keywords_match
                    image_name = os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))

                    cv2.imwrite(
                        output_path + '/' + ''.join(max(set(column_name), key=column_name.count)) + '/' + str(
                            image_name),
                        image)
                    logger = self.file_logging('keywordmatching', datas,
                                               ''.join(max(set(column_name), key=column_name.count)), fname)
                    # logger = self.file_logging('objectdetection', datas, dirs[2], fname)

                    # return (column_name, image)
                else:
                    print(str(os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))))
                    cv2.imwrite(output_path + '/' + dirs[10] + '/' + str(
                        os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))), image_np)


            elif (len(set(column_name)) == 1):

                datas = keywords_match
                image_name = os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))

                cv2.imwrite(
                    output_path + '/' + ''.join(set(column_name)) + '/' + str(image_name), image_np)
                # cv2.imwrite(
                #     output_path + '/' + ''.join(set(column_name)) + '/' + str(i) + '.jpg',
                #     image)
                logger = self.file_logging('keywordmatching', datas, ''.join(set(column_name)), fname)

            else:
                print(str(os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))))
                cv2.imwrite(output_path + '/' + dirs[10] + '/' + str(
                    os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))), image_np)
            # return (column_name, image)

        else:
            print(str(os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))))
            cv2.imwrite(output_path + '/' + dirs[10] + '/' + str(
                os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))), image_np)
            # return ([], [])

    def analyse2(self, fname, total, index):

        # model_path=r'C:\Users\mohanraj\PycharmProjects\untitled\venv\models\research\object_detection\inference_graph_med_5'
        # label_path=r'C:\Users\mohanraj\PycharmProjects\untitled\venv\models\research\object_detection\inference_graph_med_5'

        # List of the strings that is used to add correct label for each box.



        category_index = label_map_util.create_category_index(categories2)

        try:
            with detection_graph2.as_default():
                with tf.Session(graph=detection_graph2) as sess:
                    # image_np=cv2.imread(r'C:\Users\mohanraj\Desktop\IMG-20191217-WA0000 (1).jpg')
                    # image_np = cv2.imread(r'C:\Users\mohanraj\Desktop\project\Dr-Kamineni-XRay\Database Shoulder Xrays\Aequalis (Tornier).jpg')

                    image = Image.open(fname)
                    enhancer = ImageEnhance.Contrast(image)

                    enhanced_im = enhancer.enhance(1.0)
                    image = np.array(enhanced_im, np.uint8)
                    kernel = np.ones((1, 1), np.uint8)
                    img = cv2.dilate(image, kernel, iterations=1)
                    image_np = cv2.erode(img, kernel, iterations=1)
                    print('analyse2', fname)
                    image = cv2.imread(fname)

                    # img.convert('L')
                    # image_np=cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

                    # image_np= np.array(img)
                    # cv2.imwrite('temp.jpg',image_np)
                    # image_np=cv2.imread('temp.jpg')
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph2.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph2.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph2.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph2.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph2.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    # Visualization of the results of a detection.
                    # vis_util.visualize_boxes_and_labels_on_image_array(
                    #     image_np,
                    #     np.squeeze(boxes),
                    #     np.squeeze(classes).astype(np.int32),
                    #     np.squeeze(scores),
                    #     category_index,
                    #     use_normalized_coordinates=True,
                    #     line_thickness=8)
                    objects = []
                    for index, value in enumerate(classes[0]):
                        object_dict = {}
                        if scores[0  , index] > 0.6:
                            object_dict[(category_index.get(value)).get('name')] = \
                                scores[0, index]
                            objects.append(object_dict)
                    print('output', objects)
        except:
            objects = []
        if (objects != []):
            for k, v in objects[0].items():
                if ((k.split('-')[0] == 'comp') or (k == 'comp')):

                    cv2.imwrite(output_path + '/' + dirs[0] + '/' + os.path.basename(input_path + "/" + fname),
                                image)
                    cv2.imwrite(
                        output_path + '/' + dirs[0] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                        image_np)
                    datas = []
                    datas.append(k)
                    datas.append(v)
                    logger = self.file_logging('objectdetection', datas, dirs[0], fname)


                elif ((k.split('-')[0] == 'fax') or (k == 'fax')):
                    cv2.imwrite(output_path + '/' + dirs[1] + '/' + os.path.basename(input_path + "/" + fname),
                                image)
                    cv2.imwrite(
                        output_path + '/' + dirs[1] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                        image_np)
                    datas = []
                    datas.append(k)
                    datas.append(v)
                    logger = self.file_logging('objectdetection', datas, dirs[1], fname)

                elif (((k.split('-')[0] == 'demo') or (k == 'demo'))):
                    acquire = []
                    for obj in objects:
                        for f, g in obj.items():
                            if ((f == 'demo-secondaryinsurance') or (f == 'demo-primaryinsurance')):
                                acquire.append(f)
                    if (acquire != []):
                        cv2.imwrite(output_path + '/' + dirs[7] + '/' + os.path.basename(input_path + "/" + fname),
                                    image)
                        cv2.imwrite(
                            output_path + '/' + dirs[7] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                            image_np)
                        datas = []
                        datas.append(k)
                        datas.append(v)
                        logger = self.file_logging('objectdetection', datas, dirs[7], fname)

                    elif ((k == 'demo-memberdata') or (k == 'demo-memberinformation')):
                        cv2.imwrite(output_path + '/' + dirs[7] + '/' + os.path.basename(input_path + "/" + fname),
                                    image)
                        cv2.imwrite(
                            output_path + '/' + dirs[7] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                            image_np)
                        datas = []
                        datas.append(k)
                        datas.append(v)
                        logger = self.file_logging('objectdetection', datas, dirs[7], fname)

                    elif (k == 'demo-policynumber'):
                        cv2.imwrite(output_path + '/' + dirs[7] + '/' + os.path.basename(input_path + "/" + fname),
                                    image)
                        cv2.imwrite(
                            output_path + '/' + dirs[7] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                            image_np)
                        datas = []
                        datas.append(k)
                        datas.append(v)
                        logger = self.file_logging('objectdetection', datas, dirs[7], fname)

                    elif (k == 'demo-financialpolicy'):
                        cv2.imwrite(output_path + '/' + dirs[5] + '/' + os.path.basename(input_path + "/" + fname),
                                    image)
                        cv2.imwrite(
                            output_path + '/' + dirs[5] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                            image_np)
                        datas = []
                        datas.append(k)
                        datas.append(v)
                        logger = self.file_logging('objectdetection', datas, dirs[5], fname)
                    else:
                        cv2.imwrite(output_path + '/' + dirs[2] + '/' + os.path.basename(input_path + "/" + fname),
                                    image)
                        cv2.imwrite(
                            output_path + '/' + dirs[2] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                            image_np)
                        datas = []
                        datas.append(k)
                        datas.append(v)
                        logger = self.file_logging('objectdetection', datas, dirs[2], fname)

                    # elif (k == 'oxygen'):
                    #    cv2.imwrite(output_path + '/' + dirs[3] + '/' + os.path.basename(input_path + "/" + input), image)
                    #    cv2.imwrite(output_path + '/' + dirs[3] + '/' + 'log/'+ os.path.basename(input_path + "/" + input), image_np)
                elif ((k.split('-')[0] == 'pay') or (k == 'pay')):

                    if ((k == 'pay') and (v >= 0.98)):
                        cv2.imwrite(output_path + '/' + dirs[3] + '/' + os.path.basename(input_path + "/" + fname),
                                    image)
                        cv2.imwrite(
                            output_path + '/' + dirs[3] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                            image_np)
                        datas = []
                        datas.append(k)
                        datas.append(v)
                        logger = self.file_logging('objectdetection', datas, dirs[3], fname)
                    elif ((k != 'pay-merchantnumber') and (k != 'pay')):
                        cv2.imwrite(output_path + '/' + dirs[3] + '/' + os.path.basename(input_path + "/" + fname),
                                    image)
                        cv2.imwrite(
                            output_path + '/' + dirs[3] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                            image_np)
                        datas = []
                        datas.append(k)
                        datas.append(v)
                        logger = self.file_logging('objectdetection', datas, dirs[3], fname)
                    else:
                        cv2.imwrite(output_path + '/' + dirs[10] + '/' + str(
                            os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))),
                                    image)
                elif ((k.split('-')[0] == 'rx') or (k == 'rx')):
                    if ((k == 'rx-orderdate') and (v >= 0.95)):
                        cv2.imwrite(output_path + '/' + dirs[4] + '/' + os.path.basename(input_path + "/" + fname),
                                    image)
                        cv2.imwrite(
                            output_path + '/' + dirs[4] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                            image_np)
                        datas = []
                        datas.append(k)
                        datas.append(v)
                        logger = self.file_logging('objectdetection', datas, dirs[4], fname)
                    elif ((k != 'rx-orderdate')):
                        cv2.imwrite(output_path + '/' + dirs[4] + '/' + os.path.basename(input_path + "/" + fname),
                                    image)
                        cv2.imwrite(
                            output_path + '/' + dirs[4] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                            image_np)
                        datas = []
                        datas.append(k)
                        datas.append(v)
                        logger = self.file_logging('objectdetection', datas, dirs[4], fname)
                    else:
                        cv2.imwrite(output_path + '/' + dirs[10] + '/' + str(
                            os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))),
                                    image )
                   


                elif ((k.split('-')[0] == 'setup') or (k == 'setup')):
                    cv2.imwrite(output_path + '/' + dirs[5] + '/' + os.path.basename(input_path + "/" + fname),
                                image)
                    cv2.imwrite(
                        output_path + '/' + dirs[5] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                        image_np)
                    datas = []
                    datas.append(k)
                    datas.append(v)
                    logger = self.file_logging('objectdetection', datas, dirs[5], fname)

                # elif (k.split('-')[0] == 'sleep'):
                #     cv2.imwrite(output_path + '/' + dirs[6] + '/' + os.path.basename(input_path + "/" + fname),
                #                 image)
                #     cv2.imwrite(
                #         output_path + '/' + dirs[6] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                #         image_np)
                #     datas = []
                #     datas.append(k)
                #     datas.append(v)
                #     logger = self.file_logging('objectdetection', datas, dirs[6], fname)
                else:
                    # cv2.imwrite(output_path + '/' + dirs[10] + '/' + os.path.basename(input_path + "/" + fname),
                    #             image)

                    self.keyword_find(fname, total, index)
                    # cv2.imwrite(output_path + '/' + dirs[10] + '/' + str(
                    #     os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))), image_np)

            # cv2.imshow('object detection', image_np)
        # output_image = image_np
        # cv2.waitKey(0)

        elif (objects == []):

            self.keyword_find(fname, total, index)
            # cv2.imwrite(output_path + '/' + dirs[10] + '/' + str(
            #      os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))), image)

            # cv2.imwrite(output_path + '/' + dirs[7] + '/' + os.path.basename(input_path + "/" + fname),
            #                       image)
            # return ([], [])
            # cv2.imshow('object detection', image_np)

    def analyse1(self, fname, total, index):



        category_index = label_map_util.create_category_index(categories1)
        try:
           with detection_graph1.as_default():
                with tf.Session(graph=detection_graph1) as sess:
                    # image_np=cv2.imread(r'C:\Users\mohanraj\Desktop\IMG-20191217-WA0000 (1).jpg')
                    # image_np = cv2.imread(r'C:\Users\mohanraj\Desktop\project\Dr-Kamineni-XRay\Database Shoulder Xrays\Aequalis (Tornier).jpg')

                    image = Image.open(fname)
                    enhancer = ImageEnhance.Contrast(image)

                    enhanced_im = enhancer.enhance(1.0)
                    image = np.array(enhanced_im, np.uint8)
                    kernel = np.ones((1, 1), np.uint8)
                    img = cv2.dilate(image, kernel, iterations=1)
                    image_np = cv2.erode(img, kernel, iterations=1)
                    print('analyse2', fname)
                    image = cv2.imread(fname)

                    # img.convert('L')
                    # image_np=cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

                    # image_np= np.array(img)
                    # cv2.imwrite('temp.jpg',image_np)
                    # image_np=cv2.imread('temp.jpg')
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph1.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph1.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph1.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph1.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph1.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    # Visualization of the results of a detection.
                    # vis_util.visualize_boxes_and_labels_on_image_array(
                    #     image_np,
                    #     np.squeeze(boxes),
                    #     np.squeeze(classes).astype(np.int32),
                    #     np.squeeze(scores),
                    #     category_index,
                    #     use_normalized_coordinates=True,
                    #     line_thickness=8)
                    objects = []
                    for index, value in enumerate(classes[0]):
                        object_dict = {}
                        if scores[0, index] > 0.6:
                            object_dict[(category_index.get(value)).get('name')] = \
                                scores[0, index]
                            objects.append(object_dict)

        except:
            objects = []
        if (objects != []):
            for k, v in objects[0].items():

                if (k == 'comp'):
                    cv2.imwrite(output_path + '/' + dirs[0] + '/' + os.path.basename(input_path + "/" + fname),
                                image)
                    cv2.imwrite(
                        output_path + '/' + dirs[0] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                        image_np)
                    datas = []
                    datas.append(k)
                    datas.append(v)
                    logger = self.file_logging('objectdetection', datas, dirs[0], fname)


                elif ((k == 'fax') and (v >= 0.98)):
                    cv2.imwrite(output_path + '/' + dirs[1] + '/' + os.path.basename(input_path + "/" + fname),
                                image)
                    cv2.imwrite(
                        output_path + '/' + dirs[1] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                        image_np)
                    datas = []
                    datas.append(k)
                    datas.append(v)
                    logger = self.file_logging('objectdetection', datas, dirs[1], fname)

                elif ((k == 'demo')):
                    self.keyword_find(fname, total, index)
                    # cv2.imwrite(output_path + '/' + dirs[2] + '/' + os.path.basename(input_path + "/" + fname),
                    #             image)
                    # cv2.imwrite(
                    #     output_path + '/' + dirs[2] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                    #     image_np)
                    # datas = []
                    # datas.append(k)
                    # datas.append(v)
                    # logger = self.file_logging('objectdetection', datas, dirs[2], fname)

                # elif (k == 'oxygen'):
                #    cv2.imwrite(output_path + '/' + dirs[3] + '/' + os.path.basename(input_path + "/" + input), image)
                #    cv2.imwrite(output_path + '/' + dirs[3] + '/' + 'log/'+ os.path.basename(input_path + "/" + input), image_np)
                elif ((k == 'pay')):
                    cv2.imwrite(output_path + '/' + dirs[3] + '/' + os.path.basename(input_path + "/" + fname),
                                image)
                    cv2.imwrite(
                        output_path + '/' + dirs[3] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                        image_np)
                    datas = []
                    datas.append(k)
                    datas.append(v)
                    logger = self.file_logging('objectdetection', datas, dirs[3], fname)

                elif ((k == 'rx') and (v < 0.99)):

                    cv2.imwrite(output_path + '/' + dirs[4] + '/' + os.path.basename(input_path + "/" + fname),
                                image)
                    cv2.imwrite(
                        output_path + '/' + dirs[4] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                        image_np)
                    datas = []
                    datas.append(k)
                    datas.append(v)
                    logger = self.file_logging('objectdetection', datas, dirs[4], fname)

                elif (k == 'setup'):
                    cv2.imwrite(output_path + '/' + dirs[5] + '/' + os.path.basename(input_path + "/" + fname),
                                image)
                    cv2.imwrite(
                        output_path + '/' + dirs[5] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                        image_np)
                    datas = []
                    datas.append(k)
                    datas.append(v)
                    logger = self.file_logging('objectdetection', datas, dirs[5], fname)

                elif ((k == 'sleep') and (v>=0.99)):
                    cv2.imwrite(output_path + '/' + dirs[6] + '/' + os.path.basename(input_path + "/" + fname),
                                image)
                    cv2.imwrite(
                        output_path + '/' + dirs[6] + '/' + 'log/' + os.path.basename(input_path + "/" + fname),
                        image_np)
                    datas = []
                    datas.append(k)
                    datas.append(v)
                    logger = self.file_logging('objectdetection', datas, dirs[6], fname)
                    #return self.analyse2(fname, total, index)
                else:
                    # cv2.imwrite(output_path + '/' + dirs[6] + '/' + os.path.basename(input_path + "/" + fname),
                    #             image)
                    self.analyse2(fname, total, index)
                    # cv2.imwrite(output_path + '/' + dirs[10] + '/' + str(
                    #     os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))), image_np)
            # cv2.imshow('object detection', image_np)
            # output_image = image_np
            # cv2.waitKey(0)
            # return (objects[0], output_image)

        elif (objects == []):

            self.analyse2(fname, total, index)
            # cv2.imwrite(output_path + '/' + dirs[10] + '/' + str(
            #     os.path.basename(str(os.path.splitext(fname)[0]) + str(os.path.splitext(fname)[1]))), image_np)

        result = 'completed'

        object_dict = {}
        object_dict['output_list'] = {}
        object_dict['output_list']['id'] = int(index)
        object_dict['output_list']['status'] = result
        # object_dict['output_list']['move_status'] = target_foldername

        return object_dict['output_list']


@api_view(['GET', 'POST'])
def Inputparameters_list(request):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        print('GET.call')
        # print(Snippet)
        BaseSearch = ''
        serializer = BaseSearchSerializer(BaseSearch, many=True)
        print(serializer)
        d = Response(serializer.data)
        return d

    elif request.method == 'POST':

        testInstance = ParseModel()
        print('POST.call')
        print('ttt', request.data)
        # print(Snippet)
        d = request.data

        print('post data', str(d['image_path']), str(d['folder_path']))
        if ((d['image_path'] == "") and (d['folder_path'] != "")):
            for dir in dirs:
                print(output_path + '/' + dir)
                if (os.path.exists(output_path + '/' + dir) == True):
                    shutil.rmtree(output_path + '/' + dir)

                os.mkdir(output_path + '/' + dir)
                os.mkdir(output_path + '/' + dir + '/' + 'log')
            yourdata = testInstance.get_folder_path(d['folder_path'])
            '''x = []
                  da=[]
                  sc=[]
                  yx={}
                 , for d,score in yourdata:

                      yx.update({d: score})
                  print('yx',yx)'''
            # results = YourSerializer(yx, many=True).data
            print('uuuu', yourdata)
            return Response(yourdata)

        if ((d['image_path'] != "") and (d['folder_path'] == "")):
            yourdata = testInstance.analyse1(d['image_path'], d['total'], d['id'])
            print('tttt', d['total'], d['id'])

            '''x = []
                  da=[]
                  sc=[]
                  yx={}
                 , for d,score in yourdata:

                      yx.update({d: score})
                  print('yx',yx)'''
            # results = YourSerializer(yx, many=True).data
            print('uuuu', yourdata)
            return Response(yourdata)



