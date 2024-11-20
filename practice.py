import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取資料
file_path = '/Users/linjianxun/Desktop/github/vs code/1120 in class practice/1120-class_practice/Titanic Gender Submission.csv'
data = pd.read_csv(file_path)

# 印出前10筆資料
print("前10筆資料：")
print(data.head(10))

# 資料前處理：缺失值處理、標準化
# 檢查缺失值
missing_values = data.isnull().sum()
print("\n缺失值檢查：")
print(missing_values)

# 標準化 PassengerId 欄位（作為範例示範）
scaler = StandardScaler()
data['PassengerId'] = scaler.fit_transform(data[['PassengerId']])

# 分離特徵與標籤
X = data[['PassengerId']]
y = data['Survived']

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練羅吉斯回歸模型
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# 訓練決策樹模型
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_dec_tree = decision_tree.predict(X_test)

# 評估模型
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

log_reg_metrics = evaluate_model(y_test, y_pred_log_reg, 'Logistic Regression')
dec_tree_metrics = evaluate_model(y_test, y_pred_dec_tree, 'Decision Tree')

# 結果比較
results = pd.DataFrame([log_reg_metrics, dec_tree_metrics])
print("\n模型比較結果：")
print(results)

# 繪製混淆矩陣
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_test, y_pred_log_reg, 'Logistic Regression Confusion Matrix')
plot_confusion_matrix(y_test, y_pred_dec_tree, 'Decision Tree Confusion Matrix')