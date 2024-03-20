def _objective(t, dtrain, dval, early_stopping):
    params = {
        'boosting_type': t.suggest_categorical(['gbdt', 'goss']),
        'learning_rate': t.suggest_float(0.01, 0.5, log=True),
        'min_split_gain': t.suggest_float(0.00001, 2, log=True),
        'num_leaves': t.suggest_int(2, 1024, log=True),
        'max_depth': t.suggest_int(1, 15),
        'min_child_samples': t.suggest_int(2, 100, log=True),
        'bagging_freq': t.suggest_categorical([0, 1]),
        'pos_bagging_fraction': t.suggest_float(0, 1),
        'neg_bagging_fraction': t.suggest_float(0, 1),
        'reg_alpha': t.suggest_float(0.00001, 0.1, log=True),
        'reg_lambda': t.suggest_float(0.00001, 0.1, log=True),
    }
    model = lgb.train(
        **params, dtrain,
        num_boost_round=(
            4000 if early_stopping
            else trial.suggest_int('num_boost_rounds', 10, 4000)
        ),
        valid_sets=dval if early_stopping else None,
        callbacks=(
            [lgb.early_stopping(stopping_rounds=100)] if early_stopping
            else None))