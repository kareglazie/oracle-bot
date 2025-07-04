import pandas as pd

ORIGINAL_CSV = "../../data/datasets/deepseek_generated.csv"  
AUGMENTED_CSV = "../../data/datasets/deepseek_augmented.csv"
JOINED_CSV = "../../data/datasets/joined.csv"

if __name__ == "__main__":
    df1 = pd.read_csv(ORIGINAL_CSV)
    df2 = pd.read_csv(AUGMENTED_CSV)
    df2.drop(['aug_type'], axis=1, inplace=True)
    joined = pd.concat([df1, df2])
    joined.to_csv(JOINED_CSV, index=False)