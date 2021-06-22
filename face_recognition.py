import numpy as np
import cv2 as cv
import os


haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = [
    'Aaron Judge',
    'Aaron Paul',
    'Aaron Taylor-Johnson',
    'Abigail Breslin',
    'Adam Sandler',
    'Adele',
    'Adriana Barraza',
    'Adrianne Palicki',
    'Adrien Brody',
    'Akemi Darenogare',
    'Al Pacino',
    'Alan Alda'
    # 'Alan Arkin',
    # 'Alan Rickman',
    # 'Albert Brooks',
    # 'Albert Finney',
    # 'Alec Baldwin',
    # 'Alessandra Ambrosio',
    # 'Alex Pettyfer',
    # 'Alexander Skarsgard',
    # 'Alexandra Daddario',
    # 'Alexis Thorpe',
    # 'Ali Larter',
    # 'Alice Eve',
    # 'Alicia Vikander',
    # 'Alx James',
    # 'Amanda Bynes',
    # 'Amanda Crew',
    # 'Amanda Peet',
    # 'Amanda Seyfried',
    # 'Amber Heard',
    # 'Amy Adams',
    # 'Amy Ryan',
    # 'Amy Schumer',
    # 'Analeigh Tipton',
    # 'Anderson Cooper',
    # 'Andie MacDowell',
    # 'Andreea Diaconu',
    # 'Andrew Garfield',
    # 'Andrew Lincoln',
    # 'Andrew Luck',
    # 'Andy Garcia',
    # 'Andy Murray',
    # 'Andy Samberg',
    # 'Andy Serkis',
    # 'Angela Bassett',
    # 'Angelina Jolie',
    # 'Anjelica Huston',
    # 'Anna Faris',
    # 'Anna Friel',
    # 'Anna Kendrick',
    # 'Anna Paquin',
    # 'Anna Sui',
    # 'AnnaSophia Robb',
    # 'Anne Bancroft',
    # 'Anne Baxter',
    # 'Anne Hathaway',
    # 'Annette Bening',
    # 'Anthony Hopkins',
    # 'Anthony Mackie',
    # 'Anthony Perkins',
    # 'Antonio Banderas',
    # 'Armin Mueller-Stahl',
    # 'Arnold Schwarzenegger',
    # 'Art Carney',
    # 'Ashley Graham',
    # 'Ashley Greene',
    # 'Ashley Judd',
    # 'Ashton Kutcher',
    # 'Audrey Hepburn',
    # 'Audrey Tautou',
    # 'Ava Gardner',
    # 'Barabara Palvin',
    # 'Barbra Streisand',
    # 'Barry Pepper',
    # 'Ben Affleck',
    # 'Ben Foster',
    # 'Ben Johnson',
    # 'Ben Kingsley',
    # 'Ben Stiller',
    # 'Benedict Cumberbatch',
    # 'Benicio Del Toro',
    # 'Benjamin Bratt',
    # 'Berenice Bejo',
    # 'Bernie Mac',
    # 'Bette Midler',
    # 'Betty White',
    # 'Beyonce Knowles',
    # 'Bill Daley',
    # 'Bill Hader',
    # 'Bill Murray',
    # 'Bill O Reilly',
    # 'Bill Paxton',
    # 'Bill Pullman',
    # 'Bill Rancic',
    # 'Billy Bob Thornton',
]
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

DIR = os.path.dirname(__file__) +'/resources1/Adrien Brody'

img = cv.imread(os.path.join(DIR,'adrien3.jpg'))

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 6)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a loss of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
