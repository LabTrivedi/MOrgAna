import os, glob, sys
import numpy as np
from skimage.io import imread, imsave
# from scipy.ndimage import label, binary_dilation
from scipy.ndimage import binary_dilation
from skimage import measure
from skimage.measure import label


if __name__ == '__main__':
    import sys, os
    filepath = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0,os.path.join(filepath,"..",".."))

from morgana.DatasetTools import io
from morgana.DatasetTools.segmentation import io as ioSeg
from morgana.DatasetTools.morphology import computemorphology
from morgana.DatasetTools.morphology import io as ioMorph

'''
input:
- folder selected by the user (input it as a string in the script)
- user selects the _identifier string
- user selects whether unbounded objects should be removed

folder1 structure:
- img1.tif
- img2.tif
- img1_identifier.tif
- img2_identifier.tif

- select all the images and masks containing object(s)

output:
- subfolder ('splitObjects')
    - cropped images
- subsubfolder ('result_segmentation')
    - cropped masks with the same name as the images + _finalMask.tif
    - create segmentation_params.csv (save DatasetTools.segmentation.io.save_segmentation_params with default values)
    - morpho_params.json file ( generated with DatasetTools.morphology.computemorphology,
                                saved with DatasetTools.morphology.io )
- print('I am done! ;)')
'''


def parsing_images(image_folder, mask_folder, identifier_string, objects_at_border = False):
    
    # make directories if not already present
    images_output_dir = os.path.join(image_folder,'splitObjects')
    masks_output_dir = os.path.join(images_output_dir,'result_segmentation')
    if not os.path.isdir(images_output_dir):
        os.mkdir(images_output_dir)
    if not os.path.isdir(masks_output_dir):
        os.mkdir(masks_output_dir)
    
    # read images and append if only one channel is present/only greyscale image
    flist_in = io.get_image_list(image_folder, string_filter=identifier_string, 
                                              mode_filter='exclude')
    img_to_crop = []
    for f in flist_in:
        img = imread(f)
        if img.ndim == 2:
            img = np.expand_dims(img,0)
        if img.shape[-1] == np.min(img.shape):
            img = np.moveaxis(img, -1, 0)
        img_to_crop.append( img )
    
    # read masks/groundtruth
    flist_mask = io.get_image_list(mask_folder, string_filter=identifier_string, 
                                                mode_filter='include')
    
    # check that number of masks = number of images, otherwise, find missing mask
    if len(flist_in) != len(flist_mask):
        for f_in in flist_in:
            parent, filename = os.path.split(f_in)
            filename, file_extension = os.path.splitext(filename)
            mask_name = os.path.join(image_folder, filename + identifier_string + file_extension)
            if mask_name not in flist_mask:
                print('\"' + mask_name + '\" not found!')
                sys.exit('Please check that mask is present for every image in input folder!')
    
    # read and convert masks
    mask_to_crop = [ imread(f) for f in flist_mask ]
    mask_to_crop = [ g.astype(int) for g in mask_to_crop ]
    
    for i in range(len(mask_to_crop)):
        region_counter = 0
        
        # label mask
        labeled_mask, num_features = label(mask_to_crop[i], return_num=True)
        
        # for saving of cropped regions
        parent, filename = os.path.split(flist_in[i])
        filename, file_extension = os.path.splitext(filename)
        img_new_name = os.path.join(masks_output_dir, filename + "_cropped_mask" + file_extension)
        
        for region in measure.regionprops(labeled_mask):
                
            # compute coordinates of regions
            [min_row, min_col, max_row, max_col] = region.bbox
            # exclude objects at edge if required
            if not objects_at_border:
                if min_row == 0 or min_col == 0 or \
                    max_row == labeled_mask.shape[0] or max_row == labeled_mask.shape[1]:
                    # leave cropped objects_at_border in a different folder
                    border_objects_output_dir = os.path.join(images_output_dir,'objects_at_image_border')
                    if not os.path.isdir(border_objects_output_dir):
                        os.mkdir(border_objects_output_dir)
                    cropped_mask = mask_to_crop[i][min_row:max_row, min_col:max_col]
                    cropped_img = img_to_crop[i][:, min_row:max_row, min_col:max_col]
                    # save cropped regions
                    img_new_name = os.path.join(border_objects_output_dir, filename + 
                                                "_cropped%02d"%region_counter + file_extension)
                    mask_new_name = os.path.join(border_objects_output_dir, filename + 
                                                 "_cropped%02d_finalMask"%region_counter + file_extension)
                    imsave(mask_new_name, cropped_mask.astype(np.uint8))
                    imsave(img_new_name, cropped_img)
                    region_counter += 1
                    continue
            
            # crop images and masks based on coordinates of regions in mask
            cropped_mask = mask_to_crop[i][min_row:max_row, min_col:max_col]
            cropped_img = img_to_crop[i][:, min_row:max_row, min_col:max_col]
            # save cropped regions
            img_new_name = os.path.join(images_output_dir, filename + 
                                        "_cropped%02d"%region_counter + file_extension)
            mask_new_name = os.path.join(masks_output_dir, filename + 
                                         "_cropped%02d_finalMask"%region_counter + file_extension)
            imsave(mask_new_name, cropped_mask.astype(np.uint8))
            imsave(img_new_name, cropped_img)
            region_counter += 1
    
    # save parameters
    flist_cropped_images = io.get_image_list(images_output_dir)
    filenames = [os.path.split(fin)[1] for fin in flist_cropped_images]
    chosen_mask = 'user input'
    down_shape = 0.5
    thinning = smoothing = 'N.A.'
    ioSeg.save_segmentation_params(masks_output_dir,
                            filenames, chosen_mask, down_shape, thinning, smoothing)
    # compute morphological information
    # props = computemorphology.compute_morphological_info(
    #         images_output_dir, compute_meshgrid=False)
    # ioMorph.save_morpho_params(masks_output_dir, 'splitObjects', props)
    print('Done!')
    return

if __name__ == '__main__':
    
    user_input_folder = os.path.join('..','..','Images','objectsparser_testData')
    user_identifier_string = '_finalMask'
    user_objects_at_border = False
    parsing_images(user_input_folder, user_input_folder, user_identifier_string, user_objects_at_border)

    print('all run properly')

# skimage.io.imread, imsave

