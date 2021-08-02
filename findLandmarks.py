import dlib
import cv2
import copy
import numpy as np

def draw_rec(rectangles,image):

    for i in range(len(rectangles)):
        top_left = (rectangles[0].tr_corner().x,rectangles[0].bl_corner().y)
        bottom_right = (rectangles[0].bl_corner().x,rectangles[0].tr_corner().y)
        cv2.rectangle(image,top_left,bottom_right,(0,255,0),2) #img,top_left,bottom_right,color,thickness

    return image

def draw_points(points,image,is_animal=False,animal_points=[]):

    color = (0, 255, 0) 
    thickness = 3

    for i in range(68):
        if is_animal==False:
            end_point = start_point = (points.part(i).x, points.part(i).y)
        else:
            end_point = start_point = (animal_points[i][0],animal_points[i][1])

        image = cv2.line(image, start_point, end_point, color, thickness) 
    
    return image

def find_points(name):

    image = cv2.imread("images/"+name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rectangles = detector(gray)
    points = predictor(gray, rectangles[0])

    return image,rectangles,points

if __name__ == "__main__":

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    names = ["deniro.jpg","aydemirakbas.png","kimbodnia.png","panda","cat","gorilla"]

    imgs_1 = []
    imgs_12 = []
    imgs_2 = []

    for i in range(len(names)):
        if i<3:
            image,rectangles,points = find_points(names[i])
            img = draw_points(points,copy.copy(image))
            imgs_2.append(img)

            img = draw_rec(rectangles,copy.copy(image))
            imgs_12.append(img)
        else:
            points = np.load("images/"+names[i]+'_landmarks.npy')
            image = cv2.imread("images/"+names[i]+'.jpg')
            img = draw_points(points,copy.copy(image),is_animal=True,animal_points=points)
            imgs_1.append(img)

    col_2 = np.hstack(imgs_2)
    col_12 = np.hstack(imgs_12)
    col_1 = np.hstack(imgs_1)

    collage = np.vstack([col_1, col_2,col_12])
    # print(collage.shape)
    collage = cv2.resize(collage, dsize=(600, 600))

    cv2.imshow("collage",collage)
    cv2.waitKey(0)
