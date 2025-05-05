from src.data_loader import load_data
from src.models import train_and_evaluate

if __name__ == "__main__":
    X, y = load_data("data/hiring.csv")
    train_and_evaluate(X, y)
