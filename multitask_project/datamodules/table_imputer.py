import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def impute_extent_of_resection(csv_path: str, output_path: str) -> pd.DataFrame:
    """
    Điền dữ liệu thiếu trong cột 'Extent_of_Resection' bằng mô hình RandomForestClassifier.
    
    Args:
        csv_path (str): Đường dẫn đến file CSV chứa dữ liệu.
        output_path (str): Đường dẫn lưu file CSV sau khi đã điền.

    Returns:
        pd.DataFrame: DataFrame hoàn chỉnh sau khi đã điền giá trị thiếu.
    """
    df = pd.read_csv(csv_path)

    # Encode Grade nếu chưa có
    if 'Grade_Encode' not in df.columns and 'Grade' in df.columns:
        df['Grade_Encode'] = df['Grade'].map({'HGG': 1, 'SGG': 0})

    # Chia dữ liệu thành phần đầy đủ và phần thiếu
    df_full = df[df['Extent_of_Resection'].notna()].copy()
    df_miss = df[df['Extent_of_Resection'].isna()].copy()

    # Encode Extent_of_Resection thành nhãn
    le = LabelEncoder()
    df_full['Extent_Label'] = le.fit_transform(df_full['Extent_of_Resection'])

    # Các đặc trưng đầu vào
    features = [
        'Age', 'Survival_days', 'tumor_volume', 'ncr_net_volume',
        'ed_volume', 'et_volume', 'brain_volume',
        'tumor_pct', 'ncr_net_pct', 'ed_pct', 'et_pct', 'Grade_Encode'
    ]

    X_train = df_full[features]
    y_train = df_full['Extent_Label']
    X_test = df_miss[features]

    # Huấn luyện mô hình
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Đánh giá trên train
    yhat = model.predict(X_train)
    print("Training accuracy:", accuracy_score(yhat, y_train))

    # Dự đoán phần thiếu
    y_pred = model.predict(X_test)
    df.loc[df['Extent_of_Resection'].isna(), 'Extent_of_Resection'] = le.inverse_transform(y_pred)

    # Lưu kết quả
    df.to_csv(output_path, index=False)
    return df
