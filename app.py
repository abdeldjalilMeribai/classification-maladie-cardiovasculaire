from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
import os
import hashlib
import secrets
from functools import wraps
import google.generativeai as genai

# Suppression des warnings sklearn
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
app.secret_key = 'votre_cle_secrete_ici_123!'  # Changez ceci en production

# Configuration de l'API Gemini
GEMINI_API_KEY = "AIzaSyB5g7SI_laK4ZKZMqF8lmqBNFbMR7GWngQ"  # Remplacez par votre vraie clé
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-2.5-pro')

# Dictionnaire pour stocker les utilisateurs (en production, utilisez une base de données)
USERS_DB = {}
USERS_FILE = 'users_data.json'

# Charger les artefacts sauvegardés
model = joblib.load('models/heart_disease_mixmodels.pkl')
scaler = joblib.load('models/heart_disease_scaler.pkl')

# Charger les colonnes attendues par le modèle
with open('feature_columns.json', 'r') as f:
    FEATURE_COLUMNS = json.load(f)

# Fichier pour stocker les données utilisateur (en production, utilisez une vraie base de données)
USER_DATA_FILE = 'user_predictions.json'

# Fonctions d'authentification
def hash_password(password):
    """Hasher le mot de passe avec un salt"""
    salt = secrets.token_hex(16)
    salted_password = password + salt
    return hashlib.sha256(salted_password.encode()).hexdigest(), salt

def verify_password(password, hashed_password, salt):
    """Vérifier si le mot de passe correspond au hash"""
    salted_password = password + salt
    return hashlib.sha256(salted_password.encode()).hexdigest() == hashed_password

def load_users():
    """Charger les utilisateurs depuis le fichier"""
    global USERS_DB
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                USERS_DB = json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement des utilisateurs: {e}")
        USERS_DB = {}

def save_users():
    """Sauvegarder les utilisateurs dans le fichier"""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(USERS_DB, f, indent=2)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des utilisateurs: {e}")

# Charger les utilisateurs au démarrage
load_users()

def load_user_data():
    """Charger les données utilisateur depuis le fichier JSON"""
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_data(user_data):
    """Sauvegarder les données utilisateur dans le fichier JSON"""
    try:
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(user_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        return False

def get_user_id():
    """Obtenir l'identifiant utilisateur depuis la session"""
    if 'user_email' in session:
        return session['user_email']
    else:
        return None

def save_prediction_data(user_id, form_data, prediction_result):
    """Sauvegarder les données de prédiction pour un utilisateur"""
    user_data = load_user_data()
    
    if user_id not in user_data:
        user_data[user_id] = {}
    
    # Extraire les données importantes
    prediction_data = {
        'date': datetime.now().isoformat(),
        'heartRate': int(form_data.get('thalach', 72)),
        'bloodPressure': f"{int(form_data.get('trestbps', 120))}/80",
        'cholesterol': int(form_data.get('chol', 180)),
        'glucose': 90 if form_data.get('fbs') == '0' else 140,  # Estimation basée sur fbs
        'age': int(form_data.get('age', 50)),
        'prediction': int(prediction_result['prediction']),
        'probability': float(prediction_result['probability']),
        'all_form_data': dict(form_data)  # Sauvegarder toutes les données pour référence
    }
    
    user_data[user_id]['last_prediction'] = prediction_data
    
    return save_user_data(user_data)

def get_user_last_prediction(user_id):
    """Récupérer la dernière prédiction d'un utilisateur"""
    user_data = load_user_data()
    if user_id in user_data and 'last_prediction' in user_data[user_id]:
        return user_data[user_id]['last_prediction']
    return None

def preprocess_input(data):
    """
    Préprocesse les données d'entrée EXACTEMENT comme pendant l'entraînement
    """
    print(f"Données d'entrée: {data}")
    
    # 1. Créer le DataFrame avec les données brutes
    df = pd.DataFrame([data])
    print(f"DataFrame initial: {df}")
    
    # 2. Feature engineering - création de pulse_pressure
    df['pulse_pressure'] = df['trestbps'] - df['thalach']
    print(f"Après ajout pulse_pressure: {list(df.columns)}")
    
    # 3. Création des groupes d'âge
    df['age_group'] = pd.cut(df['age'], 
                           bins=[0, 40, 50, 60, 70, 100],
                           labels=['<40', '40-50', '50-60', '60-70', '70+'])
    print(f"Groupe d'âge créé: {df['age_group'].iloc[0]}")
    
    # 4. Supprimer la colonne age (comme dans l'entraînement)
    df = df.drop(columns=['age'])
    print(f"Après suppression age: {list(df.columns)}")
    
    # 5. One-Hot Encoding pour age_group avec drop_first=True
    df = pd.get_dummies(df, columns=['age_group'], drop_first=True)
    print(f"Après One-Hot age_group: {list(df.columns)}")
    
    # 6. Convertir les colonnes booléennes en entiers
    bool_columns = df.select_dtypes(include='bool').columns
    if len(bool_columns) > 0:
        df[bool_columns] = df[bool_columns].astype(int)
        print(f"Colonnes booléennes converties: {list(bool_columns)}")
    
    # 7. One-Hot Encoding pour les autres variables catégorielles avec drop_first=True
    df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
    print(f"Après One-Hot complet: {list(df.columns)}")
    
    # 8. Convertir les nouvelles colonnes booléennes en entiers
    bool_columns = df.select_dtypes(include='bool').columns
    if len(bool_columns) > 0:
        df[bool_columns] = df[bool_columns].astype(int)
        print(f"Nouvelles colonnes booléennes converties: {list(bool_columns)}")
    
    # 9. S'assurer que toutes les colonnes attendues sont présentes
    print(f"Colonnes attendues: {len(FEATURE_COLUMNS)}")
    print(f"Colonnes actuelles: {len(df.columns)}")
    
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    
    # 10. Ordonner les colonnes exactement comme pendant l'entraînement
    df = df[FEATURE_COLUMNS]
    print(f"DataFrame final avant scaling: {df.shape}")
    print(f"Colonnes finales: {list(df.columns)}")
    
    # 11. Appliquer le scaling
    scaled_array = scaler.transform(df)
    
    # 12. Recréer un DataFrame avec les noms de colonnes pour éviter les warnings
    scaled_df = pd.DataFrame(scaled_array, columns=FEATURE_COLUMNS)
    print(f"DataFrame après scaling: {scaled_df.shape}")
    
    return scaled_df

# Décorateur pour vérifier l'authentification
def login_required(f):
    """Décorateur pour vérifier si l'utilisateur est connecté"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# Routes d'authentification
@app.route('/login', methods=['GET'])
def login_page():
    """Page de connexion"""
    # Si déjà connecté, rediriger vers la home
    if 'user_email' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/register', methods=['GET'])
def register_page():
    """Page d'inscription"""
    # Si déjà connecté, rediriger vers la home
    if 'user_email' in session:
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/login', methods=['POST'])
def login():
    """Traiter la connexion"""
    try:
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            return render_template('login.html', error="Tous les champs sont requis")
        
        # Vérifier si l'utilisateur existe
        if username not in USERS_DB:
            return render_template('login.html', error="Utilisateur non trouvé")
        
        user_data = USERS_DB[username]
        
        # Vérifier le mot de passe
        if not verify_password(password, user_data['password_hash'], user_data['salt']):
            return render_template('login.html', error="Mot de passe incorrect")
        
        # Connexion réussie
        session['user_email'] = username
        session['user_name'] = username.split('@')[0] if '@' in username else username
        
        return redirect(url_for('home'))
        
    except Exception as e:
        print(f"Erreur lors de la connexion: {e}")
        return render_template('login.html', error="Erreur lors de la connexion")

@app.route('/register', methods=['POST'])
def register():
    """Traiter l'inscription"""
    try:
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not email or not password or not confirm_password:
            return render_template('register.html', error="Tous les champs sont requis")
        
        if password != confirm_password:
            return render_template('register.html', error="Les mots de passe ne correspondent pas")
        
        # Vérifier si l'utilisateur existe déjà
        if username in USERS_DB:
            return render_template('register.html', error="Ce nom d'utilisateur existe déjà")
        
        # Hasher le mot de passe
        password_hash, salt = hash_password(password)
        
        # Enregistrer l'utilisateur
        USERS_DB[username] = {
            'email': email,
            'password_hash': password_hash,
            'salt': salt,
            'created_at': datetime.now().isoformat()
        }
        
        # Sauvegarder les utilisateurs
        save_users()
        
        # Connecter automatiquement l'utilisateur
        session['user_email'] = username
        session['user_name'] = username
        
        return redirect(url_for('home'))
        
    except Exception as e:
        print(f"Erreur lors de l'inscription: {e}")
        return render_template('register.html', error="Erreur lors de l'inscription")

@app.route('/logout')
def logout():
    """Déconnexion"""
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/check_auth')
def check_auth():
    """Vérifier si l'utilisateur est authentifié"""
    # Vérifier si l'utilisateur est dans la session ET existe dans la base de données
    if 'user_email' in session and session['user_email'] in USERS_DB:
        return jsonify({'authenticated': True, 'username': session['user_email']})
    else:
        # Nettoyer la session si l'utilisateur n'existe pas
        session.clear()
        return jsonify({'authenticated': False})

# Routes protégées
@app.route('/')
@login_required
def home():
    user_id = get_user_id()
    last_prediction = get_user_last_prediction(user_id)
    return render_template('index.html', active_tab='home', last_prediction=last_prediction)

@app.route('/analysis')
@login_required
def analysis():
    # Récupérer l'utilisateur connecté
    user_id = get_user_id()
    return render_template('analysis.html', active_tab='analysis')

@app.route('/diagnostic')
@login_required
def diagnostic():
    # Récupérer tous les paramètres de l'URL
    result = {
        'prediction': request.args.get('prediction'),
        'probability': request.args.get('probability'),
        'age': request.args.get('age'),
        'sex': request.args.get('sex'),
        'cp': request.args.get('cp'),
        'trestbps': request.args.get('trestbps'),
        'chol': request.args.get('chol'),
        'fbs': request.args.get('fbs'),
        'restecg': request.args.get('restecg'),
        'thalach': request.args.get('thalach'),
        'exang': request.args.get('exang'),
        'oldpeak': request.args.get('oldpeak'),
        'slope': request.args.get('slope'),
        'ca': request.args.get('ca'),
        'thal': request.args.get('thal')
    }

    # Conversion sécurisée des valeurs numériques
    try:
        if result['probability']:
            result['probability'] = float(result['probability'])
        if result['prediction']:
            result['prediction'] = int(result['prediction'])
    except (TypeError, ValueError) as e:
        print(f"Erreur de conversion: {e}")
        result['probability'] = 0.0
        result['prediction'] = 0

    return render_template('diagnostic.html', result=result)

@app.route('/predict', methods=['GET'])
@login_required
def prediction_form():
    return render_template('prediction.html', active_tab='predict')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Récupérer et valider les données du formulaire
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                          'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        form_data = {}
        for field in required_fields:
            value = request.form.get(field)
            if value is None or value == '':
                raise ValueError(f"Champ manquant: {field}")
            try:
                form_data[field] = float(value)
            except ValueError:
                raise ValueError(f"Valeur invalide pour {field}: {value}")
        
        print(f"Données du formulaire validées: {form_data}")

        # Prétraitement des données
        processed_data = preprocess_input(form_data)
        print(f"Données prétraitées: {processed_data.shape}")

        # Prédiction avec le modèle
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]  # Probabilité de la classe positive
        
        print(f"Prédiction: {prediction}")
        print(f"Probabilité: {probability:.4f}")

        # Sauvegarder les données de prédiction pour l'utilisateur
        user_id = get_user_id()
        prediction_result = {
            'prediction': prediction,
            'probability': probability
        }
        
        save_success = save_prediction_data(user_id, form_data, prediction_result)
        if save_success:
            print(f"Données de prédiction sauvegardées pour l'utilisateur: {user_id}")
        else:
            print(f"Erreur lors de la sauvegarde des données pour l'utilisateur: {user_id}")

        # Préparer les données pour la redirection
        result_params = {
            'prediction': str(int(prediction)),
            'probability': str(round(probability, 4)),
            **{k: str(v) for k, v in form_data.items()}
        }

        print(f"Paramètres de redirection: {result_params}")

        # Redirection vers la page diagnostic avec les résultats
        return redirect(url_for('diagnostic', **result_params))

    except ValueError as ve:
        print(f"Erreur de validation: {ve}")
        return render_template('prediction.html', 
                             active_tab='predict', 
                             error=f"Erreur de validation: {str(ve)}")
    
    except Exception as e:
        print(f"Erreur inattendue lors de la prédiction: {str(e)}")
        print(f"Type d'erreur: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        return render_template('prediction.html', 
                             active_tab='predict', 
                             error=f"Erreur système: Veuillez réessayer ou contacter le support")

@app.route('/get_user_data')
@login_required
def get_user_data():
    """API pour récupérer les données de l'utilisateur connecté"""
    user_id = get_user_id()
    last_prediction = get_user_last_prediction(user_id)
    
    return jsonify({
        'user_id': user_id,
        'last_prediction': last_prediction
    })

@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html', active_tab='chatbot')


@app.route('/chatbot/send', methods=['POST'])
@login_required
def chatbot_send():
    """Route pour envoyer un message au chatbot"""
    try:
        # Récupérer le message
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message vide'}), 400
        
        # Préparer le prompt
        prompt = f"""Tu es un assistant spécialisé en santé cardiaque nommé HeartAI. 
Réponds de manière professionnelle et bienveillante en français. 
Voici la question: {user_message}"""
        
        # Appeler l'API Gemini
        response = model_gemini.generate_content(prompt)
        
        # Retourner la réponse
        return jsonify({
            'success': True,
            'response': response.text
        })
        
    except Exception as e:
        print(f"Erreur Gemini API: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Une erreur est survenue lors du traitement de votre message'
        }), 500



if __name__ == '__main__':
    print("Démarrage de l'application HeartAI...")
    print(f"Colonnes attendues par le modèle: {len(FEATURE_COLUMNS)}")
    print(f"Utilisateurs enregistrés: {len(USERS_DB)}")
    
    # Créer le fichier de données utilisateur s'il n'existe pas
    if not os.path.exists(USER_DATA_FILE):
        save_user_data({})
        print(f"Fichier de données utilisateur créé: {USER_DATA_FILE}")
    
    print("Application prête!")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)