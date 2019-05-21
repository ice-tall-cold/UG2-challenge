# Instructions on how to run the test code and get the detected files.
Add the test images dataset to this path. We assume the path name is "track2.1_test_sample". 
Then run "python test_convert_xml.py -eval_path track2.1_test_sample -pre_path results" in terminal. You will get corresponding .xml files generated in folder "results". Those .xml files can be used to run mAP_test codes and get the final mAP results.

If you want to run the test code in pycharm not in terminal, just run test_convert_xml_pycharm.py. You need to manually change the path name in the code. The root_paht is the path where stored test images dataset. You can change the root_path to the folder name of test images dataset. The result .xml files are still generated in "results" folder.

# Important Packages List
python 3.6.8  
tensorflow 1.12.0  
tensorflow-gpu 1.12.0  
tensorflow-base 1.12.0  
scikit-image 0.14.1  
scipy 1.2.1  
pycocotools 2.0  
pycoco 0.7.2  
opencv 3.4.2  
pandas 0.24.1  
numpy 1.15.4  
keras 2.2.4  
imguag 0.2.6  
cudatoolkit 9.0  
cudnn 7.3.1  
