import os
import pandas as pd
import xml.etree.ElementTree as ET
import json
import cv2
import argparse
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-i", "--image_folder", help = "path to the image folder")
parser.add_argument("-j", "--json_file", help = "path to the json file")

class JsonToXml:
    def __init__(self, json_file_path, image_folder):
        """
            To inititalize the path of json file and image folder.
            
            Input:
                json_file_path : path to the json file
                image_folder : path to the image folder
        """
        self.json_file_path = json_file_path
        self.image_folder = image_folder
        
    def json_to_csv(self):
        """
            This function converts the the categories into a dataframe so it can be later used
            for creating the xml.
            
        """
        with open(self.json_file_path,'rb') as file:
            doc = json.load(file)

        categories = doc['categories']

        categorylist = []

        for category in categories:
            categorylist.append(["none", category['id'], category['name']])

        df = pd.DataFrame(categorylist, columns =['supercategory', 'id', 'name'])

        df.to_csv('coco_categories.csv', index = False)
        
    def write_to_xml(self, image_name, image_dict, data_folder, save_folder, xml_template='pascal_voc_template.xml'):
        """
            This function does the parsing of bounding boxes for an image and create the corresponding xml for it.
            
            Input:
                image_name : name of the image
                image_dict : dictionary having key as the image_name and value as the list of bounding boxes
                             for the image
                data_folder : path to image folder
                save_folder : path where xml must be created
                xml_template : sample xml format which is required for creating the xml.
                
        """
        # get bboxes
        bboxes = image_dict[image_name]

        # read xml file
        tree = ET.parse(xml_template)
        root = tree.getroot()    

        # modify
        folder = root.find('folder')
        folder.text = 'Annotations'

        fname = root.find('filename')
        fname.text = image_name 

        src = root.find('source')
        database = src.find('database')
        database.text = 'COCO2017'


        # size
        img = cv2.imread(os.path.join(data_folder, image_name))
        h,w,d = img.shape

        size = root.find('size')
        width = size.find('width')
        width.text = str(w)
        height = size.find('height')
        height.text = str(h)
        depth = size.find('depth')
        depth.text = str(d)

        for box in bboxes:
            # append object
            obj = ET.SubElement(root, 'object')

            name = ET.SubElement(obj, 'name')
            name.text = box[0]

            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'

            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = str(0)

            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = str(0)

            bndbox = ET.SubElement(obj, 'bndbox')

            #checking the edge cases of bounding box exceeding the image itself
            xmin = ET.SubElement(bndbox, 'xmin')
            if box[1] < 0:
                xmin.text = str(0)
            else:
                xmin.text = str(box[1])

            ymin = ET.SubElement(bndbox, 'ymin')
            if box[2] < 0:
                ymin.text = str(0)
            else:
                ymin.text = str(box[2])

            xmax = ET.SubElement(bndbox, 'xmax')
            if box[3] > w:
                xmax.text = str(w)
            else:
                xmax.text = str(box[3])

            ymax = ET.SubElement(bndbox, 'ymax')
            if box[4] > h:
                ymax.text = str(h)
            else:
                ymax.text = str(box[4])

        xmin, ymin, xmax, ymax = xmin.text, ymin.text, xmax.text, ymax.text

        # save .xml to anno_path
        anno_path = os.path.join(save_folder, image_name.split('.')[0] + '.xml')
        tree.write(anno_path)
        
    def createXml(self):
        """
            This function creates the image dict which stores info of all the bounding box per image and calls
            write to xml which in-turn create the xml.
        """
        # read annotations file
        annotations_path = self.json_file_path

        # read coco category list
        df = pd.read_csv('coco_categories.csv')
        df.set_index('id', inplace=True)

        # specify image locations
        image_folder = self.image_folder

        # specify savepath - where to save .xml files
        savepath = self.image_folder
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        # read in .json format
        with open(annotations_path,'rb') as file:
            doc = json.load(file)

        # get annotations
        annotations = doc['annotations']

        # initialize dict to store bboxes for each image
        image_dict = {}

        # get image info
        image_info = doc['images']


        # loop through the annotations in the subset
        for anno in annotations:
            # get annotation for image name
            image_id = anno['image_id']
            image_name = image_info[image_id]['file_name']    
            # get category
            category = df.loc[anno['category_id']]['name']

            # add as a key to image_dict
            if not image_name in image_dict.keys():
                image_dict[image_name]=[]

            # append bounding boxes to it
            box = anno['bbox']
            # since bboxes = [xmin, ymin, width, height]:
            image_dict[image_name].append([category, box[0], box[1], box[0]+box[2], box[1]+box[3]])

        # # generate .xml files
        for image_name in image_dict.keys():

            self.write_to_xml(image_name, image_dict, image_folder, savepath)
            #print('generated for: ', image_name)

def main():
    args = parser.parse_args()

    if args.image_folder and args.json_file:
        json_file_path = args.json_file
        image_folder = args.image_folder

        json_to_xml = JsonToXml(json_file_path, image_folder)

        json_to_xml.json_to_csv()
        json_to_xml.createXml()

    else:
        print("please provide the path for both image folder and json file")

if __name__ == "__main__":
    main()
        