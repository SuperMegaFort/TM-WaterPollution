#!/bin/bash
# ==============================================================================
# WATER POLLUTION DETECTION - Siamose/Triplet Grid Search
# ==============================================================================
# Ce script lance 18 entraînements consécutifs en variant l'espace latent,
# le learning rate, et la marge de la perte Triplet.
# Chaque modèle sera sauvé dans un sous-dossier contenant ses paramètres.

# Paramètres globaux (fixés pour comparer à armes égales)
EPOCHS=10
BATCH=8
TRAIN_SCOPE="manual_crop"

# Tableaux de paramètres à tester
LATENT_DIMS=(1 16 64)
LEARNING_RATES=("1e-3" "1e-4" "1e-5")
MARGINS=(0.2 1.0)

# Calcul du nombre total d'expériences
TOTAL_EXP=$(( ${#LATENT_DIMS[@]} * ${#LEARNING_RATES[@]} * ${#MARGINS[@]} ))
CURRENT_EXP=0

echo "🚀 Démarrage du Grid Search Triplet (Total: $TOTAL_EXP expériences)"
echo "------------------------------------------------------------------"

for dim in "${LATENT_DIMS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for margin in "${MARGINS[@]}"; do
            
            ((CURRENT_EXP++))
            echo "🔥 Expérience $CURRENT_EXP / $TOTAL_EXP"
            echo "👉 Paramètres : Latent=$dim | LR=$lr | Margin=$margin"
            
            # Lancer le script python avec les paramètres dynamiques
            python pipeline/train_triplet.py \
                --train_scope $TRAIN_SCOPE \
                --epochs $EPOCHS \
                --batch $BATCH \
                --latent_dim $dim \
                --lr $lr \
                --margin $margin
                
            echo "✅ Fin de l'expérience $CURRENT_EXP"
            echo "------------------------------------------------------------------"
        done
    done
done

echo "🎉 Grid Search terminé. Tous les modèles sont sauvegardés dans models/triplet/"
