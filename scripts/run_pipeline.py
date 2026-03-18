from src.data.load_data import load_data
from src.features.build_features import build_features
from src.models.train_ranking import train_model

def main():
    df = load_data()
    df = clean_data(df)
    df = build_features(df)
    train_model(df)

if __name__ == "__main__":
    main()