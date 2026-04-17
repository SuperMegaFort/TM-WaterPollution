import os
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# CONFIGURATION GLOBALE
# ─────────────────────────────────────────────
EPOCHS = 10
BATCH = 8
TRAIN_SCOPE = "no_mask"

# PHASE 1 : Paramètres neutres/standards pour comparer les backbones
TEST_LR = "1e-4"
TEST_DROP = "0.5"
TEST_FREEZE = "" # Unfreeze

BACKBONES = ["efficientnet_v2_m", "resnet50", "densenet121", "convnext_tiny"]

# PHASE 2 : Paramètres pour le Grid Search
GRID_LRS = ["1e-3", "1e-4", "1e-5"]
GRID_DROPS = ["0.2", "0.5"]
GRID_FREEZES = ["--freeze_backbone", ""]

def run_training(backbone, lr, drop, freeze_arg):
    cmd = [
        "python", os.path.join(BASE_DIR, "pipeline", "train_grl.py"),
        "--train_scope", TRAIN_SCOPE,
        "--use_grl",
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH),
        "--backbone", backbone,
        "--lr", lr,
        "--dropout", str(drop)
    ]
    if freeze_arg:
        cmd.append(freeze_arg)
        
    print(f"\n   [EXEC] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
def get_kappa_from_summary(backbone, lr, drop, freeze_arg):
    freeze_str = "freeze" if freeze_arg else "unfreeze"
    folder_name = f"{backbone}_lr_{lr}_drp_{drop}_{freeze_str}"
    
    # Emplacement attendu (hardcodé par rapport à train_grl.py : train_all car aucune rivière précisée)
    summary_path = os.path.join(BASE_DIR, "models", "grl", TRAIN_SCOPE, "with_grl", "train_all", folder_name, "train_summary.txt")
    
    if not os.path.exists(summary_path):
        print(f"   [ERREUR] Fichier introuvable : {summary_path}")
        return 0.0
        
    with open(summary_path, "r", encoding="utf-8") as f:
        # La 1ère ligne est du genre : "Meilleur Kappa Validation : 0.8521"
        line = f.readline()
        try:
            val = float(line.split(":")[-1].strip())
            return val
        except Exception:
            return 0.0

if __name__ == "__main__":
    print("===============================================================================")
    print(" PHASE 1: ÉVALUATION DES 4 BACKBONES")
    print("===============================================================================")
    
    results = {}
    
    for b in BACKBONES:
        print(f"\n🚀 Phase 1 - Entraînement de base avec : {b}")
        try:
            # On lance l'entraînement !
            run_training(b, TEST_LR, TEST_DROP, TEST_FREEZE)
            
            # On lit le résultat de force généré
            score = get_kappa_from_summary(b, TEST_LR, TEST_DROP, TEST_FREEZE)
            results[b] = score
            print(f"✅ Terminé pour {b}. Score Kappa = {score:.4f}")
        except Exception as e:
            print(f"❌ Échec pour {b} : {e}")
            results[b] = 0.0
            
    print("\n===============================================================================")
    print(" BILAN PHASE 1")
    print("===============================================================================")
    for b, s in results.items():
        print(f" - {b:<20}: {s:.4f}")
        
    # Choisir le meilleur
    best_backbone = max(results, key=results.get)
    best_score = results[best_backbone]
    
    # Règle spéciale de l'utilisateur : si l'écart est très faible avec efficientnet, on privilégie efficientnet
    eff_score = results.get("efficientnet_v2_m", 0.0)
    
    # On définit "très peu de différence" = moins de 0.03 de Kappa, à ajuster si besoin
    THRESHOLD = 0.03 
    
    if best_backbone != "efficientnet_v2_m":
        diff = best_score - eff_score
        if diff <= THRESHOLD:
            print(f"\n⚠️ {best_backbone} est vainqueur avec un F1/Kappa de {best_score:.4f}.")
            print(f"Mais la différence avec efficientnet_v2_m ({eff_score:.4f}) est faible (Δ={diff:.4f} <= {THRESHOLD}).")
            print("👉 Application de la Règle Utilisateur : On sélectionne efficientnet_v2_m pour la suite.")
            best_backbone = "efficientnet_v2_m"
        else:
            print(f"\n🏆 {best_backbone} explose la concurrence (Δ={diff:.4f} > {THRESHOLD}). On le sélectionne pour la suite !")
    else:
        print(f"\n🏆 efficientnet_v2_m a triomphé nativement !")

    
    print("\n===============================================================================")
    print(f" PHASE 2: GRID SEARCH SUR LE VAINQUEUR ({best_backbone})")
    print("===============================================================================")
    
    total_grid_exp = len(GRID_LRS) * len(GRID_DROPS) * len(GRID_FREEZES)
    exp_cpt = 0
    
    for lr in GRID_LRS:
        for drop in GRID_DROPS:
            for frz in GRID_FREEZES:
                exp_cpt += 1
                
                # S'assurer de ne pas relancer le tout premier modèle exact de la Phase 1 s'il retombe ici
                if lr == TEST_LR and drop == TEST_DROP and frz == TEST_FREEZE:
                    print(f"\n🔥 [Exp {exp_cpt}/{total_grid_exp}] (Déjà calculé en Phase 1) -> lr={lr}, drop={drop}, freeze={frz}")
                    continue
                    
                print(f"\n🔥 [Exp {exp_cpt}/{total_grid_exp}] Recherche avec lr={lr}, drop={drop}, freeze={frz}")
                
                try:
                    run_training(best_backbone, lr, drop, frz)
                except Exception as e:
                    print(f"❌ Échec exp {exp_cpt} : {e}")
                    
    print("\n🎉 TOUT EST TERMINÉ ! Les modèles optimisés GRL sont dans : models/grl/no_mask/with_grl/train_all/")
