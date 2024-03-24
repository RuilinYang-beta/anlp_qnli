import os
import itertools

ts = ["train", "train_aug"]
ns = ["ori", "ori_char"]
ms = ["FeedForwardNN", "SimpleRNN", "BiRNN", "SimpleTransformer"]

combi = list(itertools.product(ts, ns, ms))

folders = [f"./{t}-{n}-{m}" for t, n, m in combi]

confirmation = input("Are you sure you want to remove these folders? Type 'yes' to proceed: ")

if confirmation.lower() == 'yes':
    for folder in folders:
        if os.path.exists(folder):
            os.system(f'rm -rf {folder}')
            print(f"Folder '{folder.ljust(40)}' removed successfully.")
        else:
            print(f"Folder '{folder.ljust(40)}' does not exist. Skipping removal.")
    print("All removal operations completed.")
else:
    print("Operation aborted.")
