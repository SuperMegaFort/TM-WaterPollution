#!/bin/bash
# ==============================================================================
# WATER POLLUTION DETECTION - CNN/GRL Grid Search
# ==============================================================================
# Ce script lance des entraînements consécutifs en variant l'architecture
# de base (Backbone), le learning rate, le dropout, et l'état de gel des poids.

# Paramètres globaux (fixés par l'utilisateur)
EPOCHS=10
BATCH=8
TRAIN_SCOPE="no_mask"

# Tableaux de paramètres à tester
BACKBONES=("efficientnet_v2_m" "resnet50" "densenet121" "convnext_tiny")
LEARNING_RATES=("1e-3" "1e-4" "1e-5")
DROPOUTS=("0.2" "0.5")
FREEZE_OPTIONS=("--freeze_backbone" "") # Avec et sans l'argument freeze

TOTAL_EXP=$(( ${#BACKBONES[@]} * ${#LEARNING_RATES[@]} * ${#DROPOUTS[@]} * ${#FREEZE_OPTIONS[@]} ))
CURRENT_EXP=0

echo "🚀 Démarrage du Grid Search CNN/GRL (Total: $TOTAL_EXP expériences en batch $BATCH)"
echo "------------------------------------------------------------------"

for backbone in "${BACKBONES[@]}"; do
    for drop in "${DROPOUTS[@]}"; do
        for freeze in "${FREEZE_OPTIONS[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
            
                ((CURRENT_EXP++))
                
                FREEZE_STR="Unfreeze"
                if [ "$freeze" == "--freeze_backbone" ]; then
                    FREEZE_STR="Freeze"
                fi

                echo "🔥 Expérience $CURRENT_EXP / $TOTAL_EXP"
                echo "👉 Config : $backbone | LR: $lr | Drop: $drop | $FREEZE_STR"
                
                # Exécution du script python
                # Note: --use_grl est activé explicitement
                python pipeline/train_grl.py \
                    --train_scope $TRAIN_SCOPE \
                    --use_grl \
                    --epochs $EPOCHS \
                    --batch $BATCH \
                    --backbone $backbone \
                    --lr $lr \
                    --dropout $drop \
                    $freeze
                    
                echo "✅ Fin de l'expérience $CURRENT_EXP"
                echo "------------------------------------------------------------------"
                
            done
        done
    done
done

echo "🎉 Grid Search GRL terminé. Modèles sauvegardés dans models/grl/no_mask/with_grl/train_all/"
