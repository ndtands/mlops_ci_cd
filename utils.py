from sklearn.metrics import f1_score

def print_results(model, X_train, X_dev, X_test, y_train, y_dev, y_test, selected_features_roc, selected_feat_random):
    # Sử dụng tất cả các đặc trưng
    model.fit(X_train, y_train)
    print('1. All features')
    
    print('='*40)
    print('Train:')
    y_pred = model.predict(X_train)
    print('>> F1 score:', f1_score(y_train, y_pred))
    print('Dev:')
    y_pred = model.predict(X_dev)
    print('>> F1 score:', f1_score(y_dev, y_pred))
    print('Test:')
    y_pred = model.predict(X_test)
    print('>> F1 score:', f1_score(y_test, y_pred))
    print('='*40)
    f1_all = f1_score(y_test, y_pred)
    
    # Sử dụng các đặc trưng được lựa chọn bằng phương pháp ROC_AUC
    
    model.fit(X_train[selected_features_roc], y_train)
    print('2. ROC-AUC selected fearures')
    
    print('='*40)
    print('Train:')
    y_pred = model.predict(X_train[selected_features_roc])
    print('>> F1 score:', f1_score(y_train, y_pred))
    print('Dev:')
    y_pred = model.predict(X_dev[selected_features_roc])
    print('>> F1 score:', f1_score(y_dev, y_pred))
    print('Test:')
    y_pred = model.predict(X_test[selected_features_roc])
    print('>> F1 score:', f1_score(y_test, y_pred))
    print('='*40)
    f1_roc = f1_score(y_test, y_pred)
    # Sử dụng các đặc trưng được lựa chọn bằng phương pháp Random Forest
    
    model.fit(X_train[selected_feat_random], y_train)
    print('3. Random forest selected features')
    
    print('='*40)
    print('Train:')
    y_pred = model.predict(X_train[selected_feat_random])
    print('>> F1 score:', f1_score(y_train, y_pred))
    print('Dev:')
    y_pred = model.predict(X_dev[selected_feat_random])
    print('>> F1 score:', f1_score(y_dev, y_pred))
    print('Test:')
    y_pred = model.predict(X_test[selected_feat_random])
    print('>> F1 score:', f1_score(y_test, y_pred))
    print('='*40)
    f1_random = f1_score(y_test, y_pred)
    return f1_all, f1_roc, f1_random, model