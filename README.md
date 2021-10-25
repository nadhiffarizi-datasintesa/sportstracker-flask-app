# sportstracker-flask-app #

API for Sportstracker using Flask. This repo utilizes KNN method to determine accuracy of fitness movement and also count its repetition

## Repo structure ##

* /models : Directory containing models of pushup, situp, and pullup. It is formatted as csv file. The last file in each directory is the newest version indicating the most stable performance
* /templates : Contains html template
* camera.py : Script defining VideoCamera class and all of its properties
* main.py : Main script to launch the Flask server
* pushup.py : Script initiatin and configuring pushup detection using KNN
* situp.py : Script initiatin and configuring situp detection using KNN
* pullup.py : Script initiatin and configuring pullup detection using KNN
* requirements/txt : List of python modules that are being used in this repo
