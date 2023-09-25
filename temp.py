
import pickle

data_path = "amc_dataset/RML2016/RML2016.10a_dict.pkl"

with open(data_path, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()

print(p[list(p.keys())[0]].shape)
