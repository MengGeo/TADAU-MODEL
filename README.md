# TADAU-MODEL

![](Fig1.png)


### **What is this repository for?**
TADAU-MODEL is a software for paper 《Time-Angle Dual Attention U-Net for Prestack Seismic Inversion of Gas Hydrate Reservoirs》.


### Who do I talk to?

Dajiang Meng;

a. Key Laboratory of Marine Mineral Resources, Ministry of Natural Resources, Guangzhou Marine Geological Survey, China Geological Survey, Guangzhou, China;

b. National Engineering Research Center for Gas Hydrate Exploration and Development, Guangzhou Marine Geological Survey, China Geological Survey, Guangzhou, China;

E-mail:dajiang623@163.com



### Usage

1、First, configure the Python environment using Anaconda and PyCharm. Python version 3.8 or higher, PyTorch 2.4.1, NumPy 1.24.4, Matplotlib 3.7.5 are required.

2、Download all .py files and the synthetic seismic data file 3syn_data.npz into the same folder.

3、Run train_TADAU.py and train_Unet.py separately to train the TADAU model and Unet model. The training outputs include loss values and well-trained models:
loss_TADAU.npz: Training and validation loss of the TADAU model;
TADAU.pth: Well-trained TADAU model;
loss_Unet.npz: Training and validation loss of the Unet model;
UNET.pth: Well-trained Unet model.

4、Run plot_syn_loss.py to generate the loss curve 1syn_loss.png.

5、Run out_syn_result.py to perform prediction using the trained models, and the results will be saved to inv_data_tadau&unet_test.npz.

6、Run plot_out_curves.py to visualize and compare the prediction results of the TADAU and Unet models, which will be saved as 1valid_curves.png.


### **code introduction**
	
    1. TADAU_model.py
        Source code of the TADAU model, dependent on Transform.py

    2. Unet_model.py
        Source code of the baseline Unet model

    3. train_function.py
        Loss function with semi-supervised geophysical constraints, dependent on AVO_tensor.py

    4. train_TADAU.py
        Script for training the TADAU model, including data loading, splitting and training parameter configuration

    5. train_Unet.py
        Script for training the UNet model, including data loading, splitting and training parameter configuration

    6. Transform.py
        Provides self-attention calculation functions for TADAU_model.py

    7. AVO_tensor.py
        Provides geophysical constraints for train_function.py, including functions for calculating reflection coefficients at different incident     
        angles using the Zoeppritz equation, and convolution functions corresponding to Equation (5) in the paper.

    8. plot_syn_loss.py
        Function for plotting training loss curves

    9. out_syn_result.py
        Prediction example on test data using the pre-trained TADAU and Unet models

    10. Data:
        3syn_data.npz: Theoretical synthetic seismic dataset

    11. Results:
        TADAU.pth: Well-trained TADAU model;

        UNET.pth: Well-trained Unet model;

        loss_TADAU.npz: Training and validation loss of the TADAU model;

        loss_Unet.npz: Training and validation loss of the Unet model;

        inv_data_tadau&unet_test.npz: Prediction results generated from 3syn_data.npz;

        1syn_loss.png: Comparison plot of training loss;

        1valid_curves.png: Comparison plot of prediction curves	


