import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Εισαγωγή δεδομένων
data = {
    'Model':         ['gpt']*16 + ['qwen']*16,
    'Chunk_Size':    [2000,2000,2000,2000,2000,2000,500,500,500,500,8000,8000,8000,8000,8000,8000]*2,
    'Overlap':       [200,200,0,0,1000,1000,0,0,200,200,0,0,200,200,1000,1000]*2,
    'Splitting_Type':['Recursive','Token','Recursive','Token','Recursive','Token',
                      'Recursive','Token','Recursive','Token','Recursive','Token',
                      'Recursive','Token','Recursive','Token']*2,

    #Αποτελέσματα Μετρικών
    'Faithfulness':       [0.795435897,0.63235,0.685307692,0.677825,0.774525,0.731125,
                           0.81595,0.88845,0.841725,0.856675,0.405333333,0.3794,
                           0.37975,0.303225,0.33535,0.3654,
                           0.7216,0.6005,0.62495,0.5487,0.783475,0.673475,
                           0.708675,0.810025,0.75005,0.726725,0.3626,0.288425,
                           0.2931,0.300075,0.292875,0.357675],
    'Answer_Relevance':   [0.891975,0.88674359,0.867871795,0.855692308,0.85605,0.88515,
                           0.855725,0.872125,0.858102564,0.871775,0.8585,0.883025641,
                           0.88305,0.9057,0.832425,0.893725,
                           0.794,0.78795,0.844575,0.74905,0.744575,0.76705,
                           0.767675,0.79325,0.7363,0.73515,0.79195,0.83775,
                           0.82325,0.79365,0.77725,0.753325],
    'Context_Precision':  [0.814923077,0.7604,0.7958,0.8021,0.898625,0.8643,
                           0.8893,0.87325,0.842,0.84935,0.470775,0.54655,
                           0.4916,0.4496,0.53705,0.483625,
                           0.693025,0.627075,0.5639,0.62015,0.65765,0.670825,
                           0.7083,0.7312,0.67185,0.653475,0.434725,0.4,
                           0.49375,0.40555,0.5125,0.39805],
    'Context_Recall':     [0.620461539,0.5516,0.56375,0.5622,0.792725,0.637625,
                           0.6921,0.72715,0.590725,0.6843,0.289375,0.242307692,
                           0.300175,0.26845,0.4154,0.3303,
                           0.556975,0.520825,0.52225,0.44165,0.5325,0.573325,
                           0.552925,0.566875,0.543625,0.532075,0.21875,0.17605,
                           0.231975,0.229275,0.285825,0.3251],
    'Answer_Correctness': [0.677153846,0.657263158,0.61381579,0.617891892,0.6684868421,0.6068,
                           0.618425,0.6171,0.55495,0.6218,0.6417,0.6615,
                           0.651225,0.6622,0.6758,0.637875,
                           0.5622,0.54365,0.53725,0.494175,0.477975,0.495,
                           0.4645,0.470825,0.46415,0.46965,0.50335,0.5117,
                           0.535225,0.52355,0.564925,0.49935],
}

df = pd.DataFrame(data)

#Label Encoding - Μετατροπή σε αριθμούς
df['Model_enc']          = (df['Model'] == 'gpt').astype(int)          # gpt=1, qwen=0
df['Splitting_enc']      = (df['Splitting_Type'] == 'Recursive').astype(int)  # Recursive=1

# Features & normalization 
feature_cols = ['Model_enc', 'Chunk_Size', 'Overlap', 'Splitting_enc'] #Είσοδοι
metrics      = ['Faithfulness','Answer_Relevance','Context_Precision', #Έξοδοι
                'Context_Recall','Answer_Correctness']

X_raw = df[feature_cols].values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("Features shape:", X_scaled.shape)  # (32, 4)

# Custom Dataset 
from torch.utils.data import Dataset, DataLoader

class RAGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)
    

def train_model(X, y_vals, metric_name, epochs=500, lr=0.01):
    """Εκπαιδεύει ένα μοντέλο και επιστρέφει τα βάρη (feature importance)."""

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_vals, test_size=0.2, random_state=42
    )

    train_ds = RAGDataset(X_tr, y_tr)
    test_ds  = RAGDataset(X_te, y_te)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

    model     = LinearRegression(n_features=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training 
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    # Evaluation 
    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te, dtype=torch.float32)
        y_te_t = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)
        test_loss = criterion(model(X_te_t), y_te_t).item()

    # Feature importance (absolute weights)
    weights = model.linear.weight.data.numpy().flatten()
    importance = dict(zip(feature_cols, np.abs(weights)))

    print(f"\n{'='*50}")
    print(f"  Μετρική: {metric_name}")
    print(f"  Test MSE: {test_loss:.6f}")
    print(f"  Feature Importance (|weight|):")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        bar = '█' * int(imp * 50)
        print(f"    {feat:<20} {imp:.4f}  {bar}")

    return weights, test_loss


# Εκτέλεση για όλες τις μετρικές 
results = {}
for metric in metrics:
    y = df[metric].values.astype(np.float32)
    w, loss = train_model(X_scaled, y, metric_name=metric)
    results[metric] = {'weights': w, 'test_mse': loss}


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 5, figsize=(22, 6))
feature_labels = ['Model\n', 'Chunk\nSize', 'Overlap', 'Splitting\nType']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for ax, metric in zip(axes, metrics):
    weights_raw = np.abs(results[metric]['weights'])
    weights = weights_raw / weights_raw.sum()
    bars = ax.barh(feature_labels, weights, color=colors)
    ax.set_title(metric.replace('_', '\n'), fontsize=10, fontweight='bold')
    ax.set_xlabel('|Weight|', fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(0, 1.0)

    # Annotations
    for bar, val in zip(bars, weights):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)

plt.suptitle('Feature Importance ανά RAGAS Μετρική\n(Linear Regression)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ragas_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# Ανάλυση ανά μοντέλο
for metric in metrics:
    gpt_mean  = df[df['Model']=='gpt'][metric].mean()
    qwen_mean = df[df['Model']=='qwen'][metric].mean()
    print(f"{metric:<22} GPT: {gpt_mean:.4f}  |  QWEN: {qwen_mean:.4f}  "
          f"| Diff: {abs(gpt_mean-qwen_mean):.4f}")