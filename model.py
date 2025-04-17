# ========== CONFIG ==========
TOP_N_BRANDS = 150
PRED_DAY = 1
# ============================

use_cols = ["order_date", "customer_id", "net_order_amount", "profit_x",
            "rolling_order_count_3", "rolling_order_count_7", "rolling_order_count_14",
            "average_order_amount", "total_orders_to_date", "order_frequency",
            "most_common_warehouse", "most_frequent_brand", "order_status"]

future_order_col = f"order_in_{PRED_DAY}_day"

df = pd.read_csv("customer_with_future_orders.csv", parse_dates=["order_date"])
df["placed_order"] = (df["order_status"] == "CLOSED").astype(int)
df = df.sort_values(["customer_id", "order_date"])

# Create lag features
df = df.groupby("customer_id").apply(
    lambda x: x.assign(**{col: x[col].shift(1).fillna(0)
                          for col in use_cols if col not in ["customer_id", "order_date", "order_status"]})
).reset_index(drop=True)

df = df.dropna(subset=[future_order_col])

features = ["net_order_amount", "profit_x", "rolling_order_count_3",
            "rolling_order_count_7", "rolling_order_count_14", "average_order_amount",
            "total_orders_to_date", "order_frequency", "most_common_warehouse",
            "most_frequent_brand"]

cat_cols = ["most_common_warehouse", "most_frequent_brand"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

df["quantity"] = df["net_order_amount"] / df["average_order_amount"].replace(0, 1)
avg_quantity_map = df.groupby(["customer_id", "most_frequent_brand"])['quantity'].mean().to_dict()

split_date = df["order_date"].quantile(0.90)
train = df[df["order_date"] < split_date]
test = df[df["order_date"] >= split_date]

y_train = train[future_order_col]
y_test = test[future_order_col]
X_train = train[features].copy()
X_test = test[features].copy()

# Handle imbalance with SMOTE
counter = Counter(y_train)
maj, min_ = max(counter.values()), min(counter.values())
current_ratio = min_ / maj
desired_ratio = 0.15

if current_ratio < desired_ratio:
    smote = SMOTE(sampling_strategy=desired_ratio, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
else:
    X_res, y_res = X_train, y_train

# LightGBM setup
param_grid = {
    'num_leaves': [30, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_data_in_leaf': [20, 50, 100],
    'lambda_l1': [0, 0.5, 1],
    'lambda_l2': [0, 0.5, 1]
}

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=400,
    class_weight={0: 1, 1: 3},
    random_state=42
)

scorer = make_scorer(f1_score, average='binary', pos_label=1)
clf = RandomizedSearchCV(model, param_grid, n_iter=15, scoring=scorer, cv=3, n_jobs=-1)
clf.fit(X_res, y_res, categorical_feature=cat_cols)

calibrated_clf = CalibratedClassifierCV(clf.best_estimator_, method='sigmoid', cv='prefit')
calibrated_clf.fit(X_res, y_res)

y_pred = calibrated_clf.predict(X_test)
y_prob = calibrated_clf.predict_proba(X_test)[:, 1]

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

print(f"Day {PRED_DAY} - Class 0: Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
print(f"Day {PRED_DAY} - Class 1: Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")

# === BRAND RANKING ===
cust_ids = test["customer_id"].values
brand_ids = test["most_frequent_brand"].values

avg_quantities = np.array([avg_quantity_map.get((cust, brand), 1.0)
                           for cust, brand in zip(cust_ids, brand_ids)])
expected_quantity = y_prob * avg_quantities

brand_df = pd.DataFrame({
    "brand_id": brand_ids,
    "expected_quantity": expected_quantity
})

brand_quantity_sum = brand_df.groupby("brand_id")["expected_quantity"].sum().sort_values(ascending=False)
top_brands = brand_quantity_sum.head(TOP_N_BRANDS)

brand_encoder = encoders["most_frequent_brand"]
brand_mapping = dict(zip(brand_encoder.transform(brand_encoder.classes_), brand_encoder.classes_))

print(f"\n=== Top {TOP_N_BRANDS} Brands to Stock for Day {PRED_DAY} ===")
for brand_id, expected_qty in top_brands.items():
    brand_name = brand_mapping.get(brand_id, f"Unknown_{brand_id}")
    print(f"{brand_name}: {expected_qty:.4f}")
