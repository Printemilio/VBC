import os
import uuid
import zipfile  # Importer le module zipfile au lieu de shutil

# Chemin vers le dossier contenant les images
dossier_images = "C:/Users/user/Desktop/Code/VBC/images/train2017"

# Taille cible pour le zip (1 Go = 1 * 1024 * 1024 * 1024 octets)
taille_cible_octets = 1 * 1024 * 1024 * 1024

def creer_zip_images(dossier_source, taille_cible):
    taille_actuelle = 0
    images_a_zipper = []
    nombre_dossiers = 0

    # Parcourir les images du dossier
    for racine, dossiers, fichiers in os.walk(dossier_source):
        for fichier in fichiers:
            chemin_complet = os.path.join(racine, fichier)
            taille_fichier = os.path.getsize(chemin_complet)

            # Vérifier si l'ajout de cette image dépasse la taille cible
            if taille_actuelle + taille_fichier > taille_cible:
                # Zipper les images actuelles
                nom_dossier_zip = str(uuid.uuid4())
                chemin_zip = os.path.join(dossier_source, nom_dossier_zip + '.zip')
                with zipfile.ZipFile(chemin_zip, 'w') as mon_zip:
                    for img in images_a_zipper:
                        mon_zip.write(img, os.path.basename(img))
                print(f'Dossier {nom_dossier_zip}.zip créé avec {len(images_a_zipper)} images.')
                images_a_zipper = []
                taille_actuelle = 0
                nombre_dossiers += 1

            # Ajouter l'image courante à la liste et mettre à jour la taille actuelle
            images_a_zipper.append(chemin_complet)
            taille_actuelle += taille_fichier

    # S'il reste des images non zippées
    if images_a_zipper:
        nom_dossier_zip = str(uuid.uuid4())
        chemin_zip = os.path.join(dossier_source, nom_dossier_zip + '.zip')
        with zipfile.ZipFile(chemin_zip, 'w') as mon_zip:
            for img in images_a_zipper:
                mon_zip.write(img, os.path.basename(img))
        print(f'Dossier {nom_dossier_zip}.zip créé avec {len(images_a_zipper)} images.')
        nombre_dossiers += 1

    return nombre_dossiers

# Appel de la fonction
nombre_dossiers_crees = creer_zip_images(dossier_images, taille_cible_octets)
print(f'Opération terminée. {nombre_dossiers_crees} dossiers ont été créés.')
