# importing necessary modules
import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
import numpy as np
import cv2


import label_image



#configuring folders
app = Flask(__name__, static_url_path = "/static")
UPLOAD_FOLDER = 'static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'
ALLOWED_EXTENSIONS = {'jpg', 'png', '.jpeg'}



# APP CONFIGURATIONS
app.config['SECRET_KEY'] = 'YourSecretKey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size to 2mb
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 




def allowed_file(filename):
   return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/', methods=['GET', 'POST'])
def index():
   if request.method == 'POST':
      if 'file' not in request.files:
         flash('No file attached in request')
         return redirect(request.url)
      file = request.files['file']
      if file.filename == '':
         flash('No file selected')
         return redirect(request.url)
      if file and allowed_file(file.filename):
         filename = secure_filename(file.filename)
         file.save(os.path.join(UPLOAD_FOLDER, filename))
   
      process_file(os.path.join(UPLOAD_FOLDER,filename), filename)
      
      data={ "processed_img":'static/downloads/'+filename, "uploaded_img":'static/uploads/'+filename}
      return render_template("index.html",data=data)
   return render_template('index.html')




"""Define the “process_file” function that calls the “detect_object” function which identifies the object in the image and saves it to the download folder."""
def process_file(path, filename):
   detect_object(path,filename)

def detect_object(path,filename):
   size = 2

   #loading haarcascade face detector
   classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    
   #Load image
   image = cv2.imread(path)

   # Resize the image to speed up detection
   mini = cv2.resize(image, (int(image.shape[1]/size), int(image.shape[0]/size)))

   # detect MultiScale / faces 
   faces = classifier.detectMultiScale(mini)

   # Draw rectangles around each face
   for f in faces:
      (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
      cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 4)
            
      #Save just the rectangle faces in SubRecFaces
      sub_face = image[y:y+h, x:x+w]

      FaceFileName = "test.jpg" #Saving the current image from the webcam for testing.
      cv2.imwrite(FaceFileName, sub_face)
            
      emotion = label_image.main(FaceFileName, "emotion_retrained_graph.pb", "emotion_retrained_labels.txt")# Getting the Result from the label_image file, i.e., Classification Result.
      emotion = emotion.title()# Title Case looks Stunning.
      font = cv2.FONT_HERSHEY_DUPLEX
      cv2.putText(image, emotion,(x+w,y), font, 0.5, (0,0,255), 1)

      gender = label_image.main(FaceFileName, "gender_retrained_graph.pb", "gender_retrained_labels.txt")# Getting the Result from the label_image file, i.e., Classification Result.
      gender = gender.title()# Title Case looks Stunning.
      font = cv2.FONT_HERSHEY_DUPLEX
      cv2.putText(image, gender,(x,y), font, 0.5, (255,0,0), 1)

   cv2.imwrite(f"{DOWNLOAD_FOLDER}{filename}",image) 


if __name__ == '__main__':
   app.run()