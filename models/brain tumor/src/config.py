#Data has been taken from Kaggle "https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation"
IMAGE_SIZE=256
IN_CHANNELS=3
OUT_CHANNELS=1 
INIT_FEATURES=32
DATA_PATH="./models/brain tumor/data/"
BASE_LEN = 70 #89 # len(/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_ <-!!!43.tif)
END_IMG_LEN = 4 # len(/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_43 !!!->.tif)
END_MASK_LEN = 9 # (/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_43 !!!->_mask.tif)
WEIGHT="models/brain tumor/weights/model.h5"
STEP_SIZE=4
EPOCHS=20
LR=0.13
