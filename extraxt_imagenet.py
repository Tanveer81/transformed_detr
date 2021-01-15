import os
import tarfile
import scipy.io
def extract_files_to_folder(read_path, save_path):
    save_path = '/nfs/data3/koner/data/LSVRC2015/train'
    for path, directories, files in os.walk('/nfs/data3/koner/data/LSVRC2015/train1'):
        for f in files:
            if f.endswith(".tar"):
                to_save = os.path.join(save_path,os.path.splitext(f)[0])
                if not  os.path.exists(to_save):
                    os.makedirs(to_save)
                tar = tarfile.open(os.path.join(path,f),'r')
                tar.extractall(path=to_save)
                tar.close()

def proce_mat_file(read_path):
    mat = scipy.io.loadmat(read_path)