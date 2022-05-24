This project can demonstrate where the license plates of any vehicle are located and highlight 
them with the accuracy.

At first, you would be requiring a labels.csv file and an images folder which contains all 
images to train the dataset. The images folder is attached in this google drive link.
(https://drive.google.com/drive/folders/1AMy6StUyFxR-YtiXOcBaEnpAesaiKm83?usp=sharing)
Please download all these jpeg and xml files into an images folder.

I have used jupyters notebook, and upload the files to juypter before running my codes.
Upload data_images, Model, images(after downloading the files as mentioned above from drive), 
test_images, data.yaml and labels.csv file. 



Step1:
We need to prepare the images folder into train and test. I have attached a data_images
folder which contains train and test folders which are empty. Now open and run 
presenting_data.ipynb code. This will import images to train and test folders in 
data_images.

Step2:
Open output_processing.ipynb and run the code. There is already a test image(N22.jpg) uploaded 
which will output and a video(traffic.mp4) folowing that. I have a file called test_images 
with some pictures which you can use to test my project(or download relevant images from google 
to run.).


*Please make sure your directories are correctly matched in my ipynb file

One the progran has run, the output would display something like this:

https://raw.githubusercontent.com/rk8055/Number-plate-recognition-software/main/1.JPG
https://raw.githubusercontent.com/rk8055/Number-plate-recognition-software/main/2.JPG
