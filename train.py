"""
אימון מודל PyTorch לחיזוי הישרדות בטיטאניק.

שלבים: הורדת הנתונים מ-Kaggle -> חלוקה ל-train/val/test -> pre-processing
-> אימון -> שמירת המשקולות וה-preprocessor לדיסק.

הרצה:
    python train.py

ניסויים לשיפור ביצועים (ממוצע על 5 seeds, הבחירה לפי ה-validation):

    ניסוי                     | val acc | val recall | val F1 | test acc | test recall | test F1
    --------------------------|---------|------------|--------|----------|-------------|--------
    מודל בסיס                 |  0.867  |   0.727    | 0.809  |  0.794   |    0.604    |  0.690
    1. תכונות חדשות (Title,   |  0.863  |   0.769    | 0.813  |  0.803   |    0.631    |  0.709
       FamilySize)            |         |            |        |          |             |
    2. שמירת ה-epoch הטוב     |  0.864  |   0.727    | 0.806  |  0.785   |    0.596    |  0.679
    3. משקל לשורדים (1.6)     |  0.846  |   0.754    | 0.792  |  0.748   |    0.671    |  0.669
    4. רשת גדולה (32->16)     |  0.866  |   0.727    | 0.808  |  0.784   |    0.612    |  0.683
    1+2+3  <-- נבחר           |  0.884  |   0.858    | 0.851  |  0.781   |    0.718    |  0.714
    1+2+3+4                   |  0.885  |   0.842    | 0.850  |  0.784   |    0.698    |  0.711

    מסקנה: שיפורים 1-3 פעילים בקוד. שיפור 4 לא תרם ונשאר כהערה במחלקה TitanicClassifier.
"""
import copy
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
# לפני שיפור 1:
# FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# ---- שיפור 1: תכונות חדשות - FamilySize ותואר (Title) בקידוד one-hot ----
TITLES = ['Mr', 'Mrs', 'Miss', 'Master', 'Rare']
TITLE_MAP = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Mme': 'Mrs', 'Miss': 'Miss', 'Mlle': 'Miss',
             'Ms': 'Miss', 'Master': 'Master'}  # כל תואר אחר (Dr, Rev, Col...) -> Rare
FEATURES = (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
            + ['Title_' + t for t in TITLES])

# קיבוע seed לשחזוריות התוצאות
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------- 1. הורדת הנתונים ----------
def download_data():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/train.csv'):
        import kaggle  # דורש kaggle.json תקין והצטרפות לתחרות
        kaggle.api.competition_download_files('titanic', path='./data', unzip=True)
    return pd.read_csv('data/train.csv')


# ---------- 2. Pre-processing ----------
def compute_fill_values(df):
    # ערכי המילוי מחושבים פעם אחת מה-train בלבד ונשמרים לשימוש ב-inference
    return {
        'Age': df['Age'].median(),
        'Fare': df['Fare'].median(),
        'Embarked': df['Embarked'].mode()[0],
    }


def preprocess_data(df, fill_values):
    df = df.copy()  # לא לשנות את ה-DataFrame המקורי

    # 1. מילוי ערכים חסרים לפי הערכים שחושבו על ה-train
    df['Age'] = df['Age'].fillna(fill_values['Age'])
    df['Fare'] = df['Fare'].fillna(fill_values['Fare'])
    df['Embarked'] = df['Embarked'].fillna(fill_values['Embarked'])

    # 2. המרת עמודות קטגוריאליות למספרים
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # ---- שיפור 1: תכונות חדשות ----
    # FamilySize - ה-EDA הראה שמשפחות של 2-4 שרדו יותר מנוסעים לבד או משפחות גדולות
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Title - התואר מהשם (Mr/Mrs/Miss/Master) מזהה ילדים גם כשהגיל חסר (177 נוסעים)
    title = df['Name'].str.extract(r',\s*([^\.]+)\.')[0].map(TITLE_MAP).fillna('Rare')
    for t in TITLES:
        df['Title_' + t] = (title == t).astype(int)  # one-hot: עמודה לכל תואר

    # 3. בחירת התכונות למודל
    return df[FEATURES]


# ---------- 3. Dataset ומודל ----------
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TitanicClassifier(nn.Module):
    def __init__(self, input_dim):
        super(TitanicClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

        # ---- שיפור 4 (נוסה ולא אומץ) ----
        # ניסינו רשת גדולה יותר: 32 -> 16 במקום 16 -> 8.
        # ממוצע על 5 seeds: val F1 0.808 לעומת 0.809 - אין שיפור, ולכן נשארנו
        # עם הרשת הקטנה (פחות פרמטרים = פחות סיכון ל-overfitting על 623 דוגמאות).
        # self.layer1 = nn.Linear(input_dim, 32)
        # self.layer2 = nn.Linear(32, 16)
        # self.output = nn.Linear(16, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.output(out)
        out = self.sigmoid(out)
        return out


# ---------- 4. הערכה על validation ----------
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def main():
    train_df = download_data()

    # חלוקה ל-70% train, 15% validation, 15% test (עם stratify לשמירת יחס השורדים)
    train_part, temp_part = train_test_split(
        train_df, test_size=0.3, random_state=SEED, stratify=train_df['Survived']
    )
    val_part, test_part = train_test_split(
        temp_part, test_size=0.5, random_state=SEED, stratify=temp_part['Survived']
    )

    # שמירת ה-test הגולמי (כולל Survived) לשימוש באפליקציית Streamlit
    test_part.to_csv('data/holdout_test.csv', index=False)

    # Pre-processing: ערכי המילוי וה-scaler מחושבים על ה-train בלבד
    fill_values = compute_fill_values(train_part)
    X_tr = preprocess_data(train_part, fill_values)
    X_val = preprocess_data(val_part, fill_values)
    y_tr = train_part['Survived'].values
    y_val = val_part['Survived'].values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    train_loader = DataLoader(TitanicDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TitanicDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = TitanicClassifier(input_dim=X_tr.shape[1])
    criterion = nn.BCELoss()  # משמש להערכה על ה-validation
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---- שיפור 3: משקל גבוה יותר לטעויות על שורדים ----
    # רק כ-38% שרדו, ולכן המודל נוטה לחזות "לא שרד" (Recall נמוך).
    # טעות על שורד נספרת כ-(מספר נספים / מספר שורדים) טעויות, בערך 1.6.
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    train_criterion = nn.BCELoss(reduction='none')  # loss לכל דוגמה בנפרד, כדי לשקלל
    print(f"pos_weight: {pos_weight:.2f}")

    # ---- שיפור 2: שמירת המודל מה-epoch עם ה-val loss הנמוך ביותר ----
    best_val_loss, best_state, best_epoch = float('inf'), None, 0

    # לולאת אימון
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # לפני שיפור 3:
            # loss = criterion(outputs, labels)
            weights = torch.where(labels == 1, pos_weight, 1.0)
            loss = (train_criterion(outputs, labels) * weights).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{EPOCHS}], "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    # לפני שיפור 2 נשמר המודל מה-epoch האחרון. עכשיו טוענים את הטוב ביותר:
    model.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")

    # שמירת המודל וה-preprocessor לדיסק
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/model.pt')
    joblib.dump(
        {'scaler': scaler, 'fill_values': fill_values, 'features': FEATURES},
        'models/preprocessor.joblib',
    )
    print("Model saved to models/model.pt")
    print("Preprocessor saved to models/preprocessor.joblib")
    print("Held-out test set saved to data/holdout_test.csv")


if __name__ == '__main__':
    main()
