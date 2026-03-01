# stack_fix_and_run.py
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

def nse(y,yhat):
    return 1 - ((y - yhat)**2).sum() / ((y - y.mean())**2).sum()

# load saved arrays created by advanced_pipeline.py
# paths used in advanced_pipeline: advanced_out/pred_lgb.npy, pred_transformer.npy, pred_lstm.npy
import os
out = "advanced_out"

p_lgb = np.load(os.path.join(out, "pred_lgb.npy"))    # length 6843
p_tr  = np.load(os.path.join(out, "pred_transformer.npy"))  # length 6831
p_lstm= np.load(os.path.join(out, "pred_lstm.npy"))  # length 6831

# also load truth used by LGB: advanced_out contains pred_lgb aligned with fe-val
truth_all = np.load(os.path.join(out, "pred_lgb.npy"))  # placeholder to get shape
# but we saved val_truth.npy earlier in simpler scripts; if available prefer that:
val_truth_path = os.path.join("data","processed","val_truth.npy")
if os.path.exists(val_truth_path):
    truth = np.load(val_truth_path)
else:
    # fallback: try to load from advanced_pipeline outputs: y_val_tab used inside pipeline
    # best effort: load original fe val truth if you saved it; otherwise align by trimming.
    # We'll align by trimming end of LGB truth to min_len below.
    truth = None

# Determine min length
min_len = min(len(p_lgb), len(p_tr), len(p_lstm))
print("Lengths:", len(p_lgb), len(p_tr), len(p_lstm), "=> min_len:", min_len)

# Trim to last min_len entries (align chronologically)
p_lgb_t  = p_lgb[-min_len:]
p_tr_t   = p_tr[-min_len:]
p_lstm_t = p_lstm[-min_len:]

# Load val truth if available (try common locations)
possible_truths = [
    os.path.join(out,"val_truth.npy"),
    os.path.join("data","processed","val_truth.npy"),
    os.path.join("val_truth.npy"),
]
truth_arr = None
for p in possible_truths:
    if os.path.exists(p):
        truth_arr = np.load(p)
        print("Loaded truth from:", p)
        break

if truth_arr is None:
    # fallback: ask user to provide truth file — but we can align using p_lgb's saved truth portion if present
    raise SystemExit("val_truth.npy not found. Please provide the ground-truth validation array at 'advanced_out/val_truth.npy' or 'data/processed/val_truth.npy'.")

# align truth to the same last min_len entries
truth_t = truth_arr[-min_len:]

# stack predictions
X = np.vstack([p_lgb_t, p_tr_t, p_lstm_t]).T

# meta learner
alphas = [0.1,1.0,10.0,100.0]
meta = RidgeCV(alphas=alphas, cv=3).fit(X, truth_t)
stack_pred = meta.predict(X)

print("Meta coefs:", meta.coef_)
print("Stack NSE:", nse(truth_t, stack_pred))
print("Stack RMSE:", mean_squared_error(truth_t, stack_pred)**0.5)

# Save stacked preds
np.save(os.path.join(out,"stack_pred_aligned.npy"), stack_pred)
print("Saved stacked preds to", os.path.join(out,"stack_pred_aligned.npy"))
