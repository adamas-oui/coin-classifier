from PIL import Image
import os
import re

IMG_SIZE = 175

def MergeImages(authority_name):
    my_files = os.listdir('images/' + authority_name)
    os.makedirs('training_images/' + authority_name)
    for i in range(int(len(my_files))):
        if (my_files[i].find('obv') >= 0):
            obv_file = 'images/{0}/{1}'.format(authority_name, my_files[i])
            # print(obv_file)
            matches = re.findall(r'(.*)obv(.*)', obv_file)
            rev_file = matches[0][0] + 'rev' + matches[0][1]
            # print(rev_file)
            
            obv = Image.open(obv_file)
            rev = Image.open(rev_file)
            obv = obv.resize((IMG_SIZE, IMG_SIZE))
            rev = rev.resize((IMG_SIZE, IMG_SIZE))
            new_image = Image.new(
                'RGB',(2*IMG_SIZE, IMG_SIZE), (255,255,255))
            new_image.paste(obv,(0,0))
            new_image.paste(rev,(IMG_SIZE, 0))
            
            name = re.findall(r'.*/(.*)', obv_file)[0]
            new_image.save(
                'training_images/{0}/{1}'.format(authority_name, name))

MergeImages('Alexander')
MergeImages('Antiochus')
MergeImages('Ptolemy')
MergeImages('Seleucus')
        
    