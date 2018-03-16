################################
# Authors 	GARNIER Romain #
#		GODEFROY Theo  #
################################
# README Reseau de Neurones    #
################################


Le fichier android-features.data doit être placé dans le même dossier que le .out

Le programme est constitué de 5 tests utilisant les mêmes set de données pour tester l'efficacité des paramètres suivant : les poids synaptiques, le taux d'apprentissage et l'erreur attendue.

	Test1 : 	Test témoin
				txApprentissage = 0.25
				poids = 0.77
				MSEMin = 0.01

	Test2 : 	Augmentation du taux d'apprentissage
				txApprentissage = 0.6
				poids = 0.77
				MSEMin = 0.01

	Test3 : 	Diminution des poids
				txApprentissage = 0.25
				poids = 0.45
				MSEMin = 0.01

	Test4 : 	Diminution de l'erreur attendue
				txApprentissage = 0.25
				poids = 0.77
				MSEMin = 0.005

