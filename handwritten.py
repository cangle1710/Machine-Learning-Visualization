import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

global yesCounter
global noCounter
global app
global demo



class imageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Place Image Here \n\n')
        self.setStyleSheet('''
			QLabel{
				border: 4px dotted
			}
		''')

    def setPix(self, image):
        super().setPixmap(image)


class dragAndDrop(QWidget):
    def __del__(self):
        print("HEHE")
        userInput()


    def __init__(self):
        super().__init__()
        self.resize(400, 400)  # can change this
        self.setAcceptDrops(True)

        mainL = QVBoxLayout()
        self.photoViewer = imageLabel()
        mainL.addWidget(self.photoViewer)

        self.setLayout(mainL)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()


    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()  # grabs file path

            self.set_image(file_path)
            event.accept()
            #print(file_path)


        else:
            event.ignore()

        accuracyTracker()
    def convertTo28By28(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (46, 46), interpolation = cv2.INTER_AREA)
        image = cv2.GaussianBlur(image,(3,3),0)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,3,2)
        image = cv2.GaussianBlur(image,(3,3),0)
        image = cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)

        image = image.reshape((1, 28, 28, 1))
        image = image.astype('float32') / 255

        return image


    def set_image(self, file_path):
        image = cv2.imread(file_path)
        #image = cv2.resize(image, (180, 320), interpolation = cv2.INTER_AREA)
        #cv2.imwrite(file_path, image)

        temp_img = cv2.resize(image, (443,251), interpolation = cv2.INTER_AREA)
        cv2.imwrite("temp.jpg", temp_img)
        self.photoViewer.setPix(QPixmap("temp.jpg"))

        list = []

        ####################
        ##Account Number
        image1 = image[ 200: 250, 175:450 ]
        image1 = image1[15:45, 15:45]

        list.append(self.convertTo28By28(image1))

        image2 = image[ 200: 250, 175:450 ]
        image2 = image2[15:45, 70:100]
        list.append(self.convertTo28By28(image2))

        image3 = image[ 200: 250, 175:450 ]
        image3 = image3[15:45, 113:143]
        list.append(self.convertTo28By28(image3))

        image4 = image[ 200: 250, 175:450 ]
        image4 = image4[15:45, 160:190]
        list.append(self.convertTo28By28(image4))

        image5 = image[ 200: 250, 175:450 ]
        image5 = image5[15:45, 210:240]
        list.append(self.convertTo28By28(image5))

        #Dollar amount
        image6 = image[ 350: 400, 500:700 ]
        image6 = image6[7:37, 15:45]
        list.append(self.convertTo28By28(image6))

        image7 = image[ 350: 400, 500:700 ]
        image7 = image7[7:37, 65:95]
        list.append(self.convertTo28By28(image7))

        image8 = image[ 350: 400, 500:700 ]
        image8 = image8[7:37, 115:145]
        list.append(self.convertTo28By28(image8))

        image9 = image[ 350: 400, 500:700 ]
        image9 = image9[7:37, 165:195]
        list.append(self.convertTo28By28(image9))

        #Cent Amount
        image10 = image[ 350: 400, 750:850 ]
        image10 = image10[10:38, 12:40]
        list.append(self.convertTo28By28(image10))

        image11 = image[ 350: 400, 750:850 ]
        image11 = image11[10:38, 61:89]
        list.append(self.convertTo28By28(image11))


        model = tf.keras.models.load_model('tensorflow_number_model')


        #Predict the number of the image
        predictedNumber = []
        for image in list:
            prediction1 = model.predict([image])
            predictedNumber.append(np.argmax(prediction1[0]))
            #print("YOUR NUMBER IS: ", np.argmax(prediction1[0]))
            plt.imshow(image.reshape(28,28))
            #print(prediction1)
        numb = ""
        for number in predictedNumber:
            numb = numb + str(number)
        print("")
        print("Account number: " + str(numb[:5]))
        print("Amount: $"+ str(numb[5:9]) + "." + str(numb[9:]))


        print("")
        print ("Drag another photo or close window to exit.")
        #print("Account number: ", str(predictedNumber[:5]))
        #print("Amount: $"+ str(predictedNumber[5:9]) + "." + str(predictedNumber[9:]))



def dataVisualization():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("")
    print("This is shown as a gray image. In order to train the model, we would need to normalize the data.")
    print("This is how it looks after normalizing.")
    print("Clicking the x will close the figure and bring you back to the Menu Screen")
    x_train = tf.keras.utils.normalize (x_train, axis = 1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    plt.imshow(x_train[5], cmap = plt.cm.binary)
    plt.show()

def dataVisualization2():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("")
    print("This is coming from TEST2_SalesOrder.png")
    print("This is the first box in Account Number in the Sales Order Form.")
    print("Closing the first figure will take you to the second figure.")

    img = cv2.imread('TEST2_SalesOrder.png')
    img = img[ 200: 250, 175:450 ]
    img = img[15:45, 15:45]

    plt.imshow(img)
    plt.show()

    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (46, 46), interpolation = cv2.INTER_AREA)
    image = cv2.GaussianBlur(image,(3,3),0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,3,2)
    image = cv2.GaussianBlur(image,(3,3),0)
    image = cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)

    image = image.reshape((1, 28, 28, 1))
    image = image.astype('float32') / 255


    model = tf.keras.models.load_model('tensorflow_number_model')
    IMG_SIZE=28
    xtrainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_testr=np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    print("This is what the model is reading after the image is scanned.")

    prediction = model.predict([x_testr])
    prediction1 = model.predict([image])

    accountNumberFirstInput = str(np.argmax(prediction1[0]))
    print("")
    print("MODEL TRAINING")
    print("MODEL PREDICTION: this number is ", np.argmax(prediction1[0]))
    plt.imshow(image.reshape(28,28))
    #print(prediction1)

    print("Closing the figure will take you back to the main screen.")
    plt.show()



def dataVisualization3():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    IMG_SIZE=28
    xtrainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_testr=np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    model = tf.keras.models.load_model('tensorflow_number_model')

    print("")
    print("Please wait for the Epoch to finish.")

    accuracy = model.fit(xtrainr, y_train, epochs=10, validation_split=0.3) ##training the model

    acc = accuracy.history['accuracy']
    val_acc = accuracy.history['val_accuracy']
    loss = accuracy.history['loss']
    val_loss = accuracy.history['val_loss']
    epochs = range(1,len(acc)+1)



    #accuracy
    #plt.plot(epochs, acc, 'ro', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


    print("Clicking the x will close the figure and bring you back to the Menu Screen")

def accuracyTracker ():
    global yesCounter, noCounter
    dontExit = True
    while dontExit:
        accuracyInput = input("Was all the numbers read accurately? Please type yes or no: ")
        if accuracyInput =="yes":
            yesCounter += 1
            dontExit = False
        elif accuracyInput == "no":
            noCounter += 1
            dontExit = False
        else:
            print("Not a valid choice. Please type yes or no.")
            dontExit = True

    print("Accurate reading:", yesCounter)
    print("Not accurate reading:", noCounter)


def userInput():
    global app, demo
    #app = None
    dontExit = True
    # app = QApplication(sys.argv)
    # demo = dragAndDrop()
    while dontExit:
        # if app != None:
        #     del app
        print("\n1. Start \n"
         "2. Show data visualization\n"
         "3. Exit\n")
        userInput = input("Please type 1, 2, or 3: ")
        if userInput == "1":

            demo.show() #show application
            dontExit = True
            app.exec()
            demo.photoViewer.setPix(QPixmap())
            #demo.set_image(sys.argv)
            #del app
        elif userInput =="2":
            print("\n1. See 1st visualization \n"
             "2. See 2nd visualization\n"
             "3. See 3rd visualization\n"
             "4. Return\n")
            userInput = input("Please type 1, 2, 3, or 4: ")
            if userInput == "1":
                dataVisualization()
            elif userInput == "2":
                dataVisualization2()

            elif userInput =="3":
                dataVisualization3()
            elif userInput =="4":
                #app = None
                dontExit = True
            else:
                dontExit = False
        elif userInput == "3":
            dontExit = False
            sys.exit()
        else:
            print("Not a valid choice. Please type 1 or 2: ")
            print("\n")
            dontExit = True
    # if app != None:
    #     sys.exit(app.exec())

def main ():
    global app, demo, yesCounter, noCounter
    app = QApplication(sys.argv)
    demo = dragAndDrop()
    yesCounter = 0
    noCounter = 0

    auth = False
    counter = 0
    while not auth:
        usernameInput = input("Username: ")
        passwordInput = input("Password: ")
        if usernameInput == "admin" and passwordInput == "admin":
            auth = True
        else:
            counter +=1
            tries = 3 - counter
            print("Incorrect password. You have", tries, "tries left.")
            print("")
            if counter == 3:
                print("Goodbye. Try again later.")
                exit()

    userInput()


if __name__ == "__main__":
    main()
