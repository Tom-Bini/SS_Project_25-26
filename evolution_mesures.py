import pandas as pd
import matplotlib.pyplot as plt

# 1. Chargement des données
# On suppose que le fichier 'mesure_poussee.csv' est dans le même dossier
try:
    data = pd.read_csv('mesure_poussee.csv')

    # 2. Extraction des colonnes
    # On utilise les noms exacts spécifiés : 't' et 'u_measured'
    t = data['t']
    u = data['u_measured']

    # 3. Création du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(t, u, label='Mesure de poussée', color='blue', linewidth=1.5)

    # 4. Personnalisation des axes et du titre
    plt.title('Évolution de la poussée u en fonction du temps t', fontsize=14)
    plt.xlabel('Temps t (s)', fontsize=12)
    plt.ylabel('Poussée mesurée u', fontsize=12)
    
    # Ajout d'une grille pour faciliter la lecture
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 5. Affichage du résultat
    plt.show()

except FileNotFoundError:
    print("Erreur : Le fichier 'mesure_poussee.csv' n'a pas été trouvé.")
except KeyError:
    print("Erreur : Vérifiez les noms de colonnes dans le CSV (doit être 't' et 'u_measured').")