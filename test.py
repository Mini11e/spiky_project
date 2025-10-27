import numpy as np
import random
import MAIN



if __name__ == "__main__":


    df_isi_means = [200, 3, 5]
    df = [x for x in df_isi_means if df_isi_means < np.std(df_isi_means(x)) or 0 for x in df_isi_means if df_isi_means > np.std(df_isi_means(x))]
    print(df)

   

        
