# Проверка важности признаков
import xgboost as xgb

model = xgb.Booster()
model.load_model("xgboost_model.json")

# Получение важности признаков
importance = model.get_score(importance_type="weight")

print("Важность признаков:", importance)
