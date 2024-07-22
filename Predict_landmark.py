def predict_landmark(X):
    model = load_model('holistic_model690.keras')
    prediction = model.predict(X)
    labels= ['Agua', 'Ayuda', 'Comida', 'Dar', 'Donde', 'Gracias', 'Hambre', 'Lejos', 'Nombre', 'Recibir']
    y_pred_enc = np.argmax(prediction, axis=1)
    predict = labels[y_pred_enc[0]]
    return predict
