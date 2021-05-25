import cv2
import pandas as pd
import joblib


#load model
model = joblib.load("colorDetectionModel.joblib")

#reading the image with opencv
img = cv2.imread("teletubbies.jpg")

#declaring global variables
clicked = False
r = g = b = xpos = ypos = 0


#predict colorName
def getColorName(R,G,B):
    cname = model.predict([[R,G,B]])
    #returned a numpy array of one item
    return cname[0]


#function to get x, y coordinate of mouse click
def draw_function(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global b,g,r,xpos,ypos,clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)

   


cv2.namedWindow('Detect Color')
cv2.setMouseCallback('Detect Color',draw_function)


while(1):
    cv2.imshow("Detect Color",img)

    if(clicked):
        #cv2.rectangle(image, startpoint, endpoint, color, thickness) -1 thickness fills rectangle entirely
        #(x1, y1),(x2, y2)
        cv2.rectangle(img,(495,5), (755,75), (255,255,255), -1)
        cv2.rectangle(img,(500,10), (750,70), (b,g,r), -1)

        #Creating text string to display ( Color name and RGB values )
        text = getColorName(r,g,b).upper()

        #cv2.putText(img,text,start,font(0-7), fontScale, color, thickness, lineType, (optional bottomLeft bool) )
        cv2.putText(img,text,(560,50),2,0.8,(255,255,255),2,cv2.LINE_AA)
    
    #Break the loop when user hits 'esc' key 
    if cv2.waitKey(20) & 0xFF ==27:
        break


cv2.destroyAllWindows()
