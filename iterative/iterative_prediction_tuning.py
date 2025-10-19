#!/usr/bin/env python3
"""
Script pour itérer sur la génération de variations et le nudging de prédictions.

Ce script exécute de manière itérative :
1. generate_variations.py pour créer des variations aléatoires
2. compute_prediction_diff.py pour faire du nudging entre prédictions

Usage:
    python3 iterative_prediction_tuning.py --iterations 10 --base candidate_predictions_14.json
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Exécute une commande shell et affiche la sortie."""
    print(f"\n{'='*70}")
    print(f"[ÉTAPE] {description}")
    print(f"[CMD] {' '.join(command)}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERREUR] La commande a échoué avec le code {e.returncode}")
        if e.stdout:
            print(f"[STDOUT] {e.stdout}")
        if e.stderr:
            print(f"[STDERR] {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Itère sur génération de variations et nudging de prédictions"
    )
    
    # Arguments requis
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="Nombre d'itérations à effectuer"
    )
    
    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Fichier de prédictions de base pour la première itération"
    )
    
    # Arguments optionnels pour generate_variations.py
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Nombre de variations à générer par itération (défaut: 1)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="my_variations",
        help="Répertoire de sortie pour les variations (défaut: my_variations)"
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default="iteration",
        help="Préfixe pour les fichiers de variation (défaut: iteration)"
    )
    
    # Arguments optionnels pour compute_prediction_diff.py
    parser.add_argument(
        "--nudge-factor",
        type=float,
        default=0.5,
        help="Facteur de nudging entre 0 et 1 (défaut: 0.5)"
    )
    
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="candidate_predictions",
        help="Préfixe pour les fichiers de sortie (défaut: candidate_predictions)"
    )
    
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=1,
        help="Numéro de la première itération (défaut: 1)"
    )
    
    args = parser.parse_args()
    
    # Vérifier que le fichier de base existe
    if not os.path.exists(args.base):
        print(f"[ERREUR] Le fichier de base '{args.base}' n'existe pas.")
        sys.exit(1)
    
    # Créer le répertoire de sortie s'il n'existe pas
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f"# DÉBUT DES ITÉRATIONS")
    print(f"{'#'*70}")
    print(f"Fichier de base: {args.base}")
    print(f"Nombre d'itérations: {args.iterations}")
    print(f"Facteur de nudging: {args.nudge_factor}")
    print(f"Répertoire de variations: {args.output_dir}")
    
    current_base = args.base
    
    for i in range(args.start_iteration, args.start_iteration + args.iterations):
        print(f"\n{'*'*70}")
        print(f"*** ITÉRATION {i}/{args.start_iteration + args.iterations - 1} ***")
        print(f"{'*'*70}")
        
        # Noms des fichiers pour cette itération
        variation_prefix = f"{args.prefix}_{i:03d}"
        variation_file = f"{args.output_dir}/{variation_prefix}_001.json"
        output_file = f"{args.output_prefix}_{i}.json"
        
        # Étape 1: Générer une variation
        step1_cmd = [
            "python3",
            "generate_variations.py",
            "--base", current_base,
            "--count", str(args.count),
            "--output-dir", args.output_dir,
            "--prefix", variation_prefix
        ]
        
        if not run_command(step1_cmd, f"Génération de variation {i}"):
            print(f"\n[ERREUR] Échec à l'itération {i} lors de la génération de variation")
            sys.exit(1)
        
        # Vérifier que le fichier de variation a été créé
        if not os.path.exists(variation_file):
            print(f"[ERREUR] Le fichier de variation '{variation_file}' n'a pas été créé")
            sys.exit(1)
        
        # Étape 2: Appliquer le nudging
        # Pour la première itération, on utilise le fichier de base comme "better"
        # Pour les suivantes, on utilise la sortie de l'itération précédente
        if i == args.start_iteration:
            better_file = args.base
        else:
            better_file = f"{args.output_prefix}_{i-1}.json"
        
        step2_cmd = [
            "python3",
            "compute_prediction_diff.py",
            "--better", better_file,
            "--worse", variation_file,
            "--nudge-factor", str(args.nudge_factor),
            "--output", output_file
        ]
        
        if not run_command(step2_cmd, f"Nudging pour itération {i}"):
            print(f"\n[ERREUR] Échec à l'itération {i} lors du nudging")
            sys.exit(1)
        
        # Vérifier que le fichier de sortie a été créé
        if not os.path.exists(output_file):
            print(f"[ERREUR] Le fichier de sortie '{output_file}' n'a pas été créé")
            sys.exit(1)
        
        # Mettre à jour le fichier de base pour la prochaine itération
        current_base = output_file
        
        print(f"\n[SUCCÈS] Itération {i} terminée ✓")
        print(f"  - Variation créée: {variation_file}")
        print(f"  - Sortie générée: {output_file}")
    
    print(f"\n{'#'*70}")
    print(f"# TOUTES LES ITÉRATIONS TERMINÉES AVEC SUCCÈS ✓")
    print(f"{'#'*70}")
    print(f"\nFichier final: {current_base}")


if __name__ == "__main__":
    main()
