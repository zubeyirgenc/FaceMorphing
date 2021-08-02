import cv2
import dlib
import numpy as np

def find_points(name):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread("images/"+name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rectangles = detector(gray)
    points = predictor(gray, rectangles[0])
    return image,points

def point_array_maker(points,image_shape):
    array = np.zeros((76, 2),dtype=int)
    for i in range(68):
        try:
            array[i][0] = points.part(i).x
            array[i][1] = points.part(i).y
        except:
            array[i][0] = points[i][0]
            array[i][1] = points[i][1]

    array[68] = (0,0)
    array[69] = (0,image_shape[1]/2-1)
    array[70] = (0,image_shape[1]-1)
    array[71] = (image_shape[0]/2-1,0)
    array[72] = (image_shape[0]-1,0)
    array[73] = (image_shape[0]-1,image_shape[1]-1)
    array[74] = (image_shape[0]/2-1,image_shape[1]-1)
    array[75] = (image_shape[0]-1,image_shape[1]/2-1)
    return array

def triangle_creater(image_shape,array):
    subdiv = cv2.Subdiv2D((0,0,image_shape[0],image_shape[1]))
    for i in range(76):
        dot = (array[i][0], array[i][1])
        subdiv.insert(dot)
    return subdiv.getTriangleList()

def finder(nonz):
    for i in range(len(nonz)):
        try:
            if nonz[i]==nonz[i-1]:
                return nonz[i]
        except:
            pass
        try:
            if nonz[i]==nonz[i+1]:
                return nonz[i]
        except:
            pass

def ordering(triangles,array):
    order = np.zeros((142, 3),dtype=int)
    for i,six in enumerate(triangles):
        nonz = np.nonzero(array==[six[0],six[1]])
        order[i][0] = finder(nonz[0])
        nonz = np.nonzero(array==[six[2],six[3]])
        order[i][1] = finder(nonz[0])
        nonz = np.nonzero(array==[six[4],six[5]])
        order[i][2] = finder(nonz[0])
    return order

def triangle_creater_for_second(order,array):
    triangles = np.zeros((142, 6),dtype=int)
    for i,six in enumerate(triangles):
        x1 = array[order[i][0],0]
        y1 = array[order[i][0],1]
        x2 = array[order[i][1],0]
        y2 = array[order[i][1],1]
        x3 = array[order[i][2],0]
        y3 = array[order[i][2],1]
        triangles[i] = [x1,y1,x2,y2,x3,y3]
    return triangles

def create_second_triangle(order,name="aydemirakbas.png"):
    try:
        image,points = find_points(name)
    except:
        points = np.load("images/"+name.split(".")[0]+'_landmarks.npy')
        image = cv2.imread("images/"+name.split(".")[0]+'.jpg')

    array = point_array_maker(points,image.shape)
    return triangle_creater_for_second(order,array),image

def first_triangle(name="deniro.jpg"):
    try:
        image,points = find_points(name)
    except:
        points = np.load("images/"+name.split(".")[0]+'_landmarks.npy')
        image = cv2.imread("images/"+name.split(".")[0]+'.jpg')

    array = point_array_maker(points,image.shape)
    triangles = triangle_creater(image.shape,array)
    order = ordering(triangles,array)
    return image,triangles,order

def image_triangular(triangles,image):
    for i in range(142):
        color = (0, 255, 0) 
        thickness = 1

        start_point = (triangles[i][0],triangles[i][1])
        end_point = (triangles[i][2],triangles[i][3])
        image_draw = cv2.line(image, start_point, end_point, color, thickness)
        start_point = (triangles[i][0],triangles[i][1])
        end_point = (triangles[i][4],triangles[i][5])
        image_draw = cv2.line(image_draw, start_point, end_point, color, thickness)
        start_point = (triangles[i][2],triangles[i][3])
        end_point = (triangles[i][4],triangles[i][5])
        image_draw = cv2.line(image_draw, start_point, end_point, color, thickness)
    return image_draw

def main():
    image,triangles,order = first_triangle()
    triangles_second,image_second = create_second_triangle(order)
    drawed_image_first = image_triangular(triangles,image)
    drawed_image_second = image_triangular(triangles_second,image_second)
    final_image = np.hstack([drawed_image_first,drawed_image_second])
    cv2.imshow("final_image",final_image)
    cv2.waitKey(0)

    with open('img1_triangles.npy', 'wb') as f:
        np.save(f, triangles)
    with open('img2_triangles.npy', 'wb') as f:
        np.save(f, triangles_second)

if __name__ == "__main__":
    main()