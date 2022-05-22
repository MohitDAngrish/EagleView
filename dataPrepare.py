import os
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
import json
from glob import glob
from sklearn.model_selection import train_test_split
import shutil
import argparse
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-i", "--image_folder", help = "path to the image folder")
parser.add_argument("-xe", "--xml_extension", help = "extension for xml file", default = 'xml')
parser.add_argument("-ie", "--image_extension", help = "extension for xml file")
parser.add_argument("-t", "--train_data_folder", help = "path to store train and test data after spliting ")


class DataPrepare:
    def __init__(self, image_folder, xml_extension, image_extension, train_data_folder):
        """
            To inititalize the path of json file and image folder.
            
            Input:
                image_folder : path to the image folder
                xml_extension : extension of xml file
                image_extension : extension of image file
                train_data_folder : path to store train and test data after spliting
        """
        self.image_folder = image_folder
        self.xml_extension = xml_extension
        self.image_extension = image_extension
        self.train_data_folder = train_data_folder
        #dictionary to save label count
        self.dictionary_of_labels = {}
        
    def check_equal_number_of_files(self):
        #get all the xml files
        xml = glob(self.image_folder + '/*.' + self.xml_extension)

        #get all the image files
        jpg = glob(self.image_folder + '/*.' + self.image_extension)

        print("xml files counts are : {}".format(len(xml)))
        print("jpg files counts are : {}".format(len(jpg)))
        print("sum is : {}".format(len(xml) + len(jpg)))
        return len(xml) == len(jpg)
    
    def return_not_matches(self):
        path = self.image_folder
        xml_extension = self.xml_extension
        image_extension = self.image_extension
        
        #get all the file names that are with .xml extension
        xml_file_names = [xml_file.split('/')[-1][:-4] for xml_file in glob(path + '/*.' + xml_extension)]

        #get all the file names that are with .jpg extension
        image_file_names = [image_file.split('/')[-1][:-4] for image_file in glob(path + '/*.' + image_extension)]

        return [[xml_file for xml_file in xml_file_names if xml_file not in image_file_names], [image_file for image_file in image_file_names if image_file not in xml_file_names]]
    
    def meta_data_to_json(self):
        image_label_occurence = {}

        xml_path = os.path.join(self.image_folder) + '/*.' + self.xml_extension
        xml = glob(xml_path)
        images_count = 0
        for file in xml:
            image_wise_occurrence = {}
            images_count += 1
            tree = ET.parse(file)
            root = tree.getroot()
            for objects in root.findall('object'):
                for label in objects:
                    if label.text not in image_wise_occurrence:
                        image_wise_occurrence[label.text] = 1
                    if label.text in self.dictionary_of_labels:
                        self.dictionary_of_labels[label.text] += 1
                    else:
                        self.dictionary_of_labels[label.text] = 1
                    break

            for label in image_wise_occurrence:
                if label in image_label_occurence:
                    image_label_occurence[label] += 1
                else:
                    image_label_occurence[label] = 1
        
        if not os.path.exists("meta_data"):
            os.makedirs("meta_data")
        
        with open(os.path.join("meta_data",'labels_count.json'), 'w') as fp:
            json.dump(self.dictionary_of_labels, fp, sort_keys=True, indent=4)
            
        with open(os.path.join("meta_data",'labels_image_occurence.json'), 'w') as fp:
            json.dump(image_label_occurence, fp, sort_keys=True, indent=4)
            
    def batch_copy_files(self, file_list, source_path, destination_path, image_extension, xml_extension):
        for file in file_list:
            image = (file + '.' + image_extension).split('/')[-1]
            xml = (file + '.' + xml_extension).split('/')[-1]

            if image_extension == "png":
                try:
                    img_png = Image.open(os.path.join(source_path, image))
                    rgb_im = img_png.convert('RGB')
                    rgb_im.save(os.path.join(destination_path, (file+'.jpg').split('/')[-1]))
                    shutil.copy(os.path.join(source_path, xml),
                             os.path.join(destination_path, xml))
                except:
                    print("Failed for {}".format(os.path.join(source_path, image)))
                    return
            else:

                shutil.copyfile(os.path.join(source_path, image), 
                    os.path.join(destination_path, image))
                shutil.copyfile(os.path.join(source_path, xml), 
                    os.path.join(destination_path, xml))
        return

    def create_label_map(self):
        label_file_name = "label_map.pbtxt"

        label_map_path = os.path.join(self.train_data_folder, label_file_name)

        f = open(label_map_path, "w")

        braces_l = '{'
        braces_r = '}'

        list_of_labels = list(self.dictionary_of_labels.keys())

        index = 0 
        for label in list_of_labels:
            index += 1
            if index == len(list_of_labels):
                f.write("item{}\n\tid:{}\n\tname:'{}'\n{}".format(braces_l,index, label, braces_r))
            else:
                f.write("item{}\n\tid:{}\n\tname:'{}'\n{}".format(braces_l,index, label, braces_r) + '\n')

        f.close()

    def train_test_split_for_data(self):
        # find image names
        image_files = glob(self.image_folder + '/*.' + self.image_extension)

        # remove file extension
        image_names = [name.replace("." + self.image_extension,"") for name in image_files]

        # Use scikit learn function for convenience
        train_names, test_names = train_test_split(image_names, test_size=0.3, random_state = 13)

        source_dir = self.image_folder
        test_dir = os.path.join(self.train_data_folder, 'images','test')
        train_dir = os.path.join(self.train_data_folder, 'images', 'train')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        self.batch_copy_files(train_names, source_dir, train_dir, self.image_extension, self.xml_extension)
        self.batch_copy_files(test_names, source_dir, test_dir, self.image_extension, self.xml_extension)
        self.create_label_map()

def file_missmatch(missing_images, missing_xmls):
    if not os.path.exists("missing"):
        os.makedirs("missing")
    
    image_textfile = open(os.path.join("missing", "images.txt"), "w")
    for image_name in missing_images:
        image_textfile.write(image_name + "\n")
    image_textfile.close()
    print("missing images text file stored inside missing folder.")
    
    xml_textfile = open(os.path.join("missing", "xmls.txt"), "w")
    for xml_name in missing_xmls:
        xml_textfile.write(xml_name + "\n")
    xml_textfile.close()
    print("missing xmls text file stored inside missing folder.")

def main():
    args = parser.parse_args()

    if args.image_folder and args.image_extension and args.train_data_folder:
        image_extension = args.image_extension
        image_folder = args.image_folder
        train_data_folder = args.train_data_folder
        xml_extension = args.xml_extension

        data_preparation = DataPrepare(image_folder, xml_extension, image_extension, train_data_folder)

        if data_preparation.check_equal_number_of_files():
    
            missing_images, missing_xmls = data_preparation.return_not_matches()
            
            if len(missing_images) == 0 and len(missing_xmls) == 0:
                print("No Data missmatch")
                data_preparation.meta_data_to_json()
                print("Meta details json created")
                data_preparation.train_test_split_for_data()
                print("Completed train test split!!!")
            else:
                
                print("Files missing")
                file_missmatch(missing_images, missing_xmls)
                
        else:
            
            print("The images files and xml file counts does not match")
            missing_images, missing_xmls = data_preparation.return_not_matches()
            file_missmatch(missing_images, missing_xmls)

    else:
        print("please provide the path for both image folder, train data folder and extension for image file")

if __name__ == "__main__":
    main()  