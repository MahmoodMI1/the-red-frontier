# the-red-frontier
In a journey to understand the huge amount of data of Mar's surface. This convolutional neural networks embarks on a journey to identify different terrains

#data
Using the HiRISE 3.2

#dimentions
    -each landmark is 227 x 227 pixels
    -64,947 total seperate images
#content
    1.map-proj-v3_2 / directory with cropped landmark images
    2.labels-map-proj-v3_2.txt: two columns: filename, class_id (seperated by a space)
    3.labels-map-proj-v3_2_train_val_test.txt, used for traind models
    (filename, class_id)
    4.landmarks_map-proj-v3_2_classmap.csv
    (class_id -> it's semantic names) map
# plan 
 - first make the Hirise class similar to the classes in the torch.utils.data classes
 - add the __getitem()__ and __len__()



 # learned
 - to store images in vars: use (form PIL import Image)
 -
    