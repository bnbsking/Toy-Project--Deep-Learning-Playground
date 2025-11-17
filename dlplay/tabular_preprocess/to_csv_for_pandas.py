import pandas as pd


def boston_housing(path: str) -> pd.DataFrame:
    """
    Csv is liked:
     0.00632  18.00   2.310  0  0.5380  6.5750  65.20  4.0900   1  296.0  15.30 396.90   4.98  24.00
     0.02731   0.00   7.070  0  0.4690  6.4210  78.90  4.9671   2  242.0  17.80 396.90   9.14  21.60
     0.02729   0.00   7.070  0  0.4690  7.1850  61.10  4.9671   2  242.0  17.80 392.83   4.03  34.70
    shape: (506, 14)
    """
    col = ["ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV", "price"]
    df = pd.read_csv(path, delim_whitespace=True)
    df.columns = col
    return df


if __name__ == "__main__":
    df = boston_housing("/app/_examples/boston_housing/dataset/housing.csv")
    print(df.shape)
    print(df.head())