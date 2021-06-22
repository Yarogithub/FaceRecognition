#pylint:disable=no-member

import os
import cv2 as cv
import numpy as np

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
DIR = os.path.dirname(__file__) +'/resources'
# print(DIR)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')
# np.save('features.npy', features)
# np.save('labels.npy', labels)
