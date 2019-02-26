from functions import *

if __name__ == '__main__':
    No1 = np.ones(shape=(128*128, 3))
    la = lambertian_attenuation(3)
    ans = normal_harmonics(No1.T, la)
    print ans.shape
    print ans[0]