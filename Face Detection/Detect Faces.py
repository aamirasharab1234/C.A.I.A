import cv2 #import the logic of this script

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert Normal Image to grayscale
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)#Detecting Features In Classifier
    coords = []
    for [x, y, w, h] in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2,LINE_AA) #Put Text on img, The Text To Put, Position of the text, font, font size, font color, Font Thickness, line
        coords = [x, y, w, h]
    
    return coords, img

def detect (img, faceCascade):
    color = ("blue":(52, 152, 219), "red":(192, 57, 43), "green":(46, 204, 113) ) #Dictionary of different colors ("Color Name": (RGB CODE))
    coords, img = draw_boundary(img, faceCascade, 1.1, 10, color['blue'])


video_capture = cv2.VideoCapture(-1) #The camera to use for video capture (-1 for external camera e.g phone with droidcam, 0 for built in default webcam managed by OS)

while True:
    _, img = video_capture.read() #Take video from camera and set image data to 'img' variable
    cv2,imshow("face detection", img) #Name of the window, Name of variable that contains image data
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()