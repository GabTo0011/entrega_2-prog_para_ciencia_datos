import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocesamiento(df):
    """
    Función para limpiar y transformar el dataset de cáncer de piel.
    """
    df_clean = df.copy()
    
    # 1. Manejo de Valores Nulos (Basado en tu punto 2 del EDA)
    # Imputamos variables numéricas con la mediana para evitar sesgo por outliers
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
    # Imputamos variables categóricas con la moda
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # 2. Transformación de Categorías (Encoding)
    # Identificar todas las columnas categóricas
    all_cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Separar en binarias (Sí/No) y multi-categoría
    bin_cols = []
    multi_cat_cols = []
    
    for col in all_cat_cols:
        unique_vals = df_clean[col].unique()
        if len(unique_vals) <= 2:
            bin_cols.append(col)
        else:
            multi_cat_cols.append(col)
    
    # Label Encoding para binarias
    for col in bin_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])

    # One-Hot Encoding para variables con múltiples categorías
    # Esto evita que el modelo piense que una categoría es "mayor" que otra numéricamente
    if multi_cat_cols:
        df_clean = pd.get_dummies(df_clean, columns=multi_cat_cols, drop_first=True)

    # 3. Escalamiento de Datos (StandardScaler)
    # Vital para que la Edad (65) no domine sobre el Tamaño (0.5 cm)
    scaler = StandardScaler()
    # No escalamos el target (Antecedentes personales)
    features_to_scale = df_clean.drop('Antecedentes personales de cáncer', axis=1).columns
    df_clean[features_to_scale] = scaler.fit_transform(df_clean[features_to_scale])
    
    return df_clean
