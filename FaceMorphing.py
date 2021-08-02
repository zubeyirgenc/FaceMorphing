import delaunayTriangulation
import cv2
import numpy as np
import moviepy.editor as mpy
from tqdm import tqdm


def make_homogeneus(triangle):
    homogeneus = np.array([triangle[::2],
                           triangle[1::2],
                           [1,1,1]]) #(C) in linear algebra many function require square matrix and so on this provide us most near and bigger
                                     #square matrix with using 6 triangle coordinate it returns 3x3 matrix is that [[y0,y1,y3],[x0,x1,x2],[1,1,1]]
                                     #and in this formation y's standing with each other in row0, x's standing with each other in row1 for easy usage
    return homogeneus

def calc_transform(triangle1,triangle2):
    source = make_homogeneus(triangle1).T
    target = triangle2
    Mtx = np.array([np.concatenate((source[0],np.zeros(3))),  #[y0,x,1,0,0,0]
                    np.concatenate((np.zeros(3),source[0])),  #[0,0,0,y0,x0,1]
                    np.concatenate((source[1],np.zeros(3))),  #[y1,x1,1,0,0,0]
                    np.concatenate((np.zeros(3),source[1])),  #[0,0,0,y1,x1,1]
                    np.concatenate((source[2],np.zeros(3))),  #[y2,x2,1,0,0,0]
                    np.concatenate((np.zeros(3),source[2]))]) #[0,0,0,y2,x2,1] #(D) it returns a 6x6 matrix from homogenous matrix of source triangle
                                                              #for be able to taking inverse and multiply to target matrix that 6x1

    coefs = np.matmul(np.linalg.pinv(Mtx),target) #(E) np.linalg.pinv function give us inverse of Mtx matrix which generating from source triangle
                                                  #with using SVD and when multiply target triangle it give us coefficient matrix that using in source
                                                  #triangule convert to target shape in form 6x1

    Transforms = np.array([coefs[:3],coefs[3:],[0,0,1]]) #(F) because of y's and x's seperate in rest of the code in there put y's to row0 and x's to row1

    return Transforms #3x3

def vectorised_Bilinear(coordinates,target_img,size):
    coordinates[0] = np.clip(coordinates[0],0,size[0]-1) #get all values of coordinates[0] in range of 0 and
    coordinates[1] = np.clip(coordinates[1],0,size[1]-1) # size[0]-1 for ifthere are point that out of 400x400
    lower = np.floor(coordinates).astype(np.uint32) #round all values to most near small integer
    upper = np.ceil(coordinates).astype(np.uint32)  #round all values to most near big integer

    error = coordinates - lower
    resindual = 1 - error

    top_left = np.multiply(np.multiply(resindual[0],resindual[1]).reshape(
        coordinates.shape[1],1),target_img[lower[0],lower[1],:])
    top_right = np.multiply(np.multiply(resindual[0],error[1]).reshape(
        coordinates.shape[1],1),target_img[lower[0],upper[1],:])
    bot_left = np.multiply(np.multiply(error[0],resindual[1]).reshape(
        coordinates.shape[1],1),target_img[upper[0],lower[1],:])
    bot_right = np.multiply(np.multiply(error[0],error[1]).reshape(
        coordinates.shape[1],1),target_img[upper[0],upper[1],:]) #(G) while in transforming of a triangle's pixels to other location if there are some
                                                                 #overflowing with in floating point(because we round all of them) this multiply with
                                                                 #error values with pixel values for not be empty between triangles. and for not be 
                                                                 #unlogical values, in this way find mean of the intermediate pixels with using floating
                                                                 #values
                                                                 # target_img[upper[0],upper[1],:] do the transformation that calculated from other function
                                                                 #in upper[0] and upper[1] includes spesific triangles and ineer points coordinates

    return np.uint8(np.round(top_left + top_right + bot_left + bot_right)) #(H) sum all calculated points because the error and residual arrays elements
                                                                           #summation give exactly one and with this way overlfowing points too calculating
                                                                           #and in ineer points there are no error
                                                                           #round it to integer because pixel values is only can integer
                                                                           #the np.uint8 function give us same array but different type. This type(uint8)
                                                                           #most efficient and true way for keeping a image

def image_morph(image1,image2,triangles1,triangles2,transforms,t):
    inter_image_1 = np.zeros(image1.shape).astype(np.uint8)
    inter_image_2 = np.zeros(image2.shape).astype(np.uint8)

    for i in range(len(transforms)):
        homo_inter_tri = (1-t)*make_homogeneus(triangles1[i]) + t*make_homogeneus(triangles2[i]) #(I) it give us average triangle of same ID triangles
        polygon_mask = np.zeros(image1.shape[:2],dtype=np.uint8)                                 #with different multiplier. same like others, this is
                                                                                                 #too in homogenious format

        cv2.fillPoly(polygon_mask,[np.int32(np.round(homo_inter_tri[1::-1,:].T))],color=255) #(J) it draw a triangle in black image whick kept by polygon_mask
                                                                                             #the triangles indicating by homo_inter_tri arrays reverse order
                                                                                             #start from the 1.element and transposes of this array.
                                                                                             #fillyPoly function make white ineer of the triangle

        seg = np.where(polygon_mask==255) #(K) this line return an array that kept 2D array. One dimension keeps x's of the ineer point of the triangle
                                          #because this pixels value is 255 another keeps y's

        mask_points = np.vstack((seg[0],seg[1],np.ones(len(seg[0])))) #(L) this line make 3xn matris from seg array. First row keep x's of ineer point of 
                                                                      #triangle, second keep y' and third is zeros. because it will multiply by 3x3

        inter_tri = homo_inter_tri[:2].flatten(order="F") #(M) it transform 3x3 average triangles to 1x6 because make fit to calculate_transform functon

        inter_to_img1 = calc_transform(inter_tri,   triangles1[i])
        inter_to_img2 = calc_transform(inter_tri,   triangles2[i])

        mapped_to_img1 = np.matmul(inter_to_img1,mask_points)[:-1] #(N) this line generate an array that required go to first
                                                                   #images triangle from average triangle. matmul make matris multiply.
                                                                   #And line order it reverse. with using mapped_to_img1 in image array it
                                                                   #return the all points of the triangle

        mapped_to_img2 = np.matmul(inter_to_img2,mask_points)[:-1]

        inter_image_1[seg[0],seg[1],:] = vectorised_Bilinear(
            mapped_to_img1,image1,inter_image_1.shape) #(O) the first images, ith triangles transformed pixels come from this function and it
                                                       #put in our black image's, which initiallize at head of the function, calculated average triangle
        inter_image_2[seg[0],seg[1],:] = vectorised_Bilinear(
            mapped_to_img2,image2,inter_image_2.shape)

    result = (1-t)*inter_image_1 + t*inter_image_2 #(P) it put one on the top of the other with multiplying with multiplier t for finding all
                                                   #intermediate passing images. Result is an image.
    return result.astype(np.uint8)

if __name__ == "__main__":
    try:
        img1_triangles = np.load("img1_triangles.npy")
        img2_triangles = np.load("img2_triangles.npy")
        img1 = cv2.imread("images/deniro.jpg")
        img2 = cv2.imread("images/aydemirakbas.png")
    except:
        img1,img1_triangles,order = delaunayTriangulation.first_triangle(name="deniro.png")
        img2_triangles,img2 = delaunayTriangulation.create_second_triangle(order,name="aydemirakbas.jpg")

    img1_triangles = img1_triangles[:,[1,0,3,2,5,4]] #y's to x, x's to y
    img2_triangles = img2_triangles[:,[1,0,3,2,5,4]]

    Transforms = np.zeros((len(img1_triangles),3,3))

    for i in range(len(img1_triangles)):
        source = img1_triangles[i]
        target = img2_triangles[i]
        Transforms[i] = calc_transform(source,target) #(A) sends to transform matrix finder function named calc_transform
                                                      #every same ID triangle points and find everyones transformation matrix
                                                      #but after this line Transforms matrix using only len(Transforms) i do not
                                                      #understand why this for workin on there because it has same length with img1_triangles    
    morphs = []
    for t in tqdm(np.arange(0,1.0001,0.02), unit =" sample", desc ="Low Passing... "): #(B) the arrange function return a array that start from 0 and finish to 1 it provide us 51 image from first
                                       #image to second image as step by step
        morphs.append(image_morph(img1,img2,img1_triangles,
            img2_triangles,Transforms,t)[:,:,::-1])

    clip = mpy.ImageSequenceClip(morphs, fps=5)
    clip.write_videofile('deniro_akbas.mp4', codec='libx264')