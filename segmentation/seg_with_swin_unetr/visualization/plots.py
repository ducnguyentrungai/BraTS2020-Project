import matplotlib.pyplot as plt
import pandas as pd



if __name__ == "__main__":
    df = pd.read_csv("/work/cuc.buithi/brats_challenge/code/segmentation/seg_with_swin_unetr/logs/train_logs.csv")
    loss = df['Loss']
    epochs = df['Epoch']
    plt.plot(loss)
    plt.savefig('loss.png')
    plt.show()
    print(df.head(10))
