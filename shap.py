from pathlib import Path
import json, numpy as np, pandas as pd, torch, shap, matplotlib.pyplot as plt
from Model import create_model

DATA_DIR = Path("Data/processed")
MODEL_DIR = Path("Models/DLASS_PRP")
SHAP_DIR = MODEL_DIR / "SHAP"
SHAP_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(DATA_DIR / "selected_features.json") as f:
    selected_features = json.load(f)
with open(DATA_DIR / "target_classes.json") as f:
    target_mapping = json.load(f)

train_df = pd.read_csv(DATA_DIR / "train_processed.csv")
test_df = pd.read_csv(DATA_DIR / "test_processed.csv")
X_train = train_df[selected_features]
X_test = test_df[selected_features]

model = create_model(num_features=len(selected_features), num_classes=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE))
model.eval()

def predict_proba(X):
    X = np.asarray(X, dtype=np.float32)
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return torch.softmax(model(x), dim=1).cpu().numpy()

background = X_train.sample(min(40, len(X_train)), random_state=42)
explain_data = X_test.copy()

explainer = shap.KernelExplainer(predict_proba, background.values)
shap_values = explainer.shap_values(explain_data.values, nsamples=100)

if isinstance(shap_values, list):
    shap_array = np.stack(shap_values, axis=-1)
else:
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3 and shap_array.shape[1] == 3 and shap_array.shape[2] == len(selected_features):
        shap_array = np.transpose(shap_array, (0, 2, 1))

class_names = [target_mapping[str(i)] for i in range(3)]
mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 2))
importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Mean_Absolute_SHAP": mean_abs_shap
}).sort_values("Mean_Absolute_SHAP", ascending=False)

importance_df.to_csv(SHAP_DIR / "global_feature_importance.csv", index=False)
pd.DataFrame(shap_array.reshape(-1, len(selected_features))).to_csv(
    SHAP_DIR / "shap_values.csv", index=False
)

plt.figure()
shap.summary_plot(shap_array, explain_data, feature_names=selected_features, show=False)
plt.savefig(SHAP_DIR / "summary_plot.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_array, explain_data, feature_names=selected_features, plot_type="bar", show=False)
plt.savefig(SHAP_DIR / "bar_plot.png", dpi=300, bbox_inches="tight")
plt.close()

for i, class_name in enumerate(class_names):
    plt.figure()
    shap.summary_plot(
        shap_array[:, :, i], explain_data,
        feature_names=selected_features, show=False
    )
    plt.savefig(SHAP_DIR / f"summary_{class_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

importance_df.to_csv(SHAP_DIR / "feature_ranking.csv", index=False)
print("SHAP analysis completed.")
print("Results saved to:", SHAP_DIR.resolve())
display(importance_df)