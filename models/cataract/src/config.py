oc_path="models/cataract/data/full_df.csv"
oc_img_path="models/cataract/data/ODIR-5K/ODIR-5K/Training Images"
device='cuda:0' if torch.cuda.is_available() else 'cpu'
IMG_SIZE=256
BATCH=64
EPOCHS=20