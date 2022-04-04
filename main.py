import cv2
import numpy as np
import Preprocess
import math

#lấy ngưỡng động
ADAPTIVE_THRESH_BLOCK_SIZE = 19 
ADAPTIVE_THRESH_WEIGHT = 9  

Min_char_area = 0.015
Max_char_area = 0.06

Min_char = 0.01
Max_char = 0.09

Min_ratio_char = 0.25
Max_ratio_char = 0.7

max_size_plate = 18000
min_size_plate = 5000

#
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

tongframe = 0
biensotimthay = 0

#Load KNN model
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32) 

# reshape numpy array to 1d, necessary to pass to call to train
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

# instantiate KNN object
kNearest = cv2.ml.KNearest_create()                   
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

#Đọc video
cap = cv2.VideoCapture('gen.mp4')
while(cap.isOpened()):

    # tiền xử lý ảnh
    ret, img = cap.read()
    tongframe = tongframe + 1
    #img = cv2.resize(img, None, fx=0.5, fy=0.5) 
    imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
    
    #Tách biên bằng canny
    canny_image = cv2.Canny(imgThreshplate,250,255) 
    kernel = np.ones((3,3), np.uint8)
    
    #tăng sharp cho egde (Phép nở). để biên canny chỗ nào bị đứt thì nó liền lại để vẽ contour
    dilated_image = cv2.dilate(canny_image,kernel,iterations=1) 

    # lọc vùng biển số 
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    #Lấy 10 contours có diện tích lớn nhất
    contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10] 
    screenCnt = []
    for c in contours:
        
        #Tính chu vi
        peri = cv2.arcLength(c, True) 
        
        #Làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
        approx = cv2.approxPolyDP(c, 0.06 * peri, True) 
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w/h
        if (len(approx) == 4) and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
            screenCnt.append(approx)
    if screenCnt is None:
        detected = 0
        print ("No plate detected")
    else:
        detected = 1

    if detected == 1:
        n=1
        for screenCnt in screenCnt:

            ################## Tính góc xoay###############
            (x1,y1) = screenCnt[0,0]
            (x2,y2) = screenCnt[1,0]
            (x3,y3) = screenCnt[2,0]
            (x4,y4) = screenCnt[3,0]
            array = [[x1, y1], [x2,y2], [x3,y3], [x4,y4]]
            sorted_array = array.sort(reverse=True, key=lambda x:x[1])
            (x1,y1) = array[0]
            (x2,y2) = array[1]

            doi = abs(y1 - y2)
            ke = abs (x1 - x2)
            angle = math.atan(doi/ke) * (180.0 / math.pi) 

            # Masking the part other than the number plate
            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
  
            # Now crop
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            roi = img[topx:bottomx + 1, topy:bottomy + 1]
            imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1]

            ptPlateCenter = (bottomx - topx)/2, (bottomy - topy)/2

            if x1 < x2:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
            else:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

            roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx ))
            imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx ))

            roi = cv2.resize(roi,(0,0),fx = 3, fy = 3)
            imgThresh = cv2.resize(imgThresh,(0,0),fx = 3, fy = 3)


            #Tiền xử lý biển số
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            thre_mor = cv2.morphologyEx(imgThresh,cv2.MORPH_DILATE,kerel3)
            cont,hier = cv2.findContours(thre_mor,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            #Phân đoạn kí tự
            char_x_ind = {}
            char_x = []
            height, width,_ = roi.shape
            roiarea = height*width
            
            #print ("roiarea",roiarea)
            for ind,cnt in enumerate(cont) :
                area = cv2.contourArea(cnt)
                (x,y,w,h) = cv2.boundingRect(cont[ind])
                ratiochar = w/h
                if (Min_char*roiarea < area < Max_char*roiarea) and ( 0.25 < ratiochar < 0.7):
                    
                    #Sử dụng để dù cho trùng x vẫn vẽ được
                    if x in char_x: 
                        x = x + 1
                    char_x.append(x)    
                    char_x_ind[x] = ind

            # Nhận diện kí tự và in ra số xe
            if len(char_x) in range (7,10):
                cv2.drawContours(img, [screenCnt], -1, (0,255, 0), 3)

                char_x = sorted(char_x) 
                strFinalString = ""
                first_line = ""
                second_line = ""

                for i in char_x:
                    (x,y,w,h) = cv2.boundingRect(cont[char_x_ind[i]])
                    cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
                    
                    # cắt kí tự ra khỏi hình
                    imgROI = thre_mor[y:y+h,x:x+w] 
                        
                    # resize lại hình ảnh
                    imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)) 
                    
                    # đưa hình ảnh về mảng 1 chiều
                    #cHUYỂN ảnh thành ma trận có 1 hàng và số cột là tổng số điểm ảnh trong đó    
                    npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      
                    
                    
                    # chuyển mảng về dạng float
                    npaROIResized = np.float32(npaROIResized)     
                     
                    # call KNN function find_nearest; neigh_resp là hàng xóm  
                    _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 3)
                    
                    # Lấy mã ASCII của kí tự đang xét    
                    strCurrentChar = str(chr(int(npaResults[0][0])))  
                    cv2.putText(roi, strCurrentChar, (x, y+50),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
                    
                    # Biển số 1 hay 2 hàng
                    if (y < height/3): 
                        first_line = first_line + strCurrentChar
                    else:
                        second_line = second_line + strCurrentChar
                
                strFinalString = first_line + second_line   
                print ("\n License Plate " +str(n)+ " is: " + first_line + " - " + second_line + "\n")
                cv2.putText(img, strFinalString, (topy ,topx),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
                n = n + 1
                biensotimthay = biensotimthay + 1

                cv2.imshow("Xac Dinh BKS",cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))

                imgcopy = cv2.resize(img, None, fx=0.5, fy=0.5)
                cv2.imshow('Video', imgcopy)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                     break

cap.release()
cv2.destroyAllWindows()



