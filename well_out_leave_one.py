# Almacenar resultados de la validación cruzada
results = []

# Para cada pozo como conjunto de prueba
for test_well in wells:
    print(f"\n=== Validación con {test_well} como prueba ===")
    
    # Dividir en entrenamiento y prueba
    X_train = Df_SC[Df_SC['well'] != test_well][['P_IMPEDANCE', 'S_IMPEDANCE', 'Vp/Vs']].values
    
    y_train = Df_SC[Df_SC['well'] != test_well]['Cluster'].values.ravel()
    
    X_test = Df_SC[Df_SC['well'] == test_well][['P_IMPEDANCE', 'S_IMPEDANCE', 'Vp/Vs']].values
    
    y_test = Df_SC[Df_SC['well'] == test_well]['Cluster'].values.ravel()
    
    # Identificar pozos de entrenamiento
    train_wells = [w for w in wells if w != test_well]
    print(f"Entrenando con: {', '.join(train_wells)}")
    print(f"Evaluando en: {test_well}")
    
    # Entrenar el modelo
    model = LogisticRegression(random_state=16, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Coeficientes del modelo
    print(f"\nCoeficientes del modelo:")
    print(f"Intercepto: {model.intercept_}")
    for i, feature_name in enumerate(['P_IMPEDANCE', 'S_IMPEDANCE', 'Vp/Vs']):
        print(f"Coeficiente para {feature_name}: {model.coef_[0][i]:.4f}")
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # Probabilidades para la clase Arena
    
    # Evaluar modelo
    print("\nResultados para el conjunto de prueba:")
    acc = evaluate_model(y_test, y_pred, y_proba[:, 1])
    
    # Guardar resultados
    results.append({
        'test_well': test_well,
        'train_wells': train_wells,
        'accuracy': acc,
        'intercept': model.intercept_[0],
        'coefficients': {
            'P_IMPEDANCE': model.coef_[0][0],
            'S_IMPEDANCE': model.coef_[0][1],
            'Vp/Vs': model.coef_[0][2]
        }
    })