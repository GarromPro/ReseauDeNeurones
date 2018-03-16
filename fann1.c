#include <stdio.h>
#include <stdlib.h>
#include <fann.h>

/*
Nb neurones cachés
Tx apprentissage
Poids
MSE min
*/


void verification(struct fann* ann, struct fann_train_data* test_data);
void apprentissage(struct fann* ann, struct fann_train_data* learn_data, struct fann_train_data* validation_data, float MSEMin);


int main() 
{
	struct fann* ann;
	struct fann_train_data *train_data, *test_data, *learn_data, *validation_data;
	int length = 0;

	//Parametres d'apprentissage
	float txApprentissage, poids, MSEMin;

	//Récupération des données
	train_data = fann_read_train_from_file("./android-features.data");
	fann_shuffle_train_data(train_data);
	length = fann_length_train_data(train_data);
	train_data = fann_subset_train_data(train_data, 0, length*.2);
	length = fann_length_train_data(train_data);


	//Répartition des données en 3 jeux de données distincts
	learn_data = fann_subset_train_data(train_data, 0, length*.6);	
	validation_data = fann_subset_train_data(train_data, length*.6, length*.2);
	test_data = fann_subset_train_data(train_data, length*.8, length*.2);



	//////////
	//TEST1///
	//TEMOIN//
	//////////

	printf("//////\n");
	printf("TEST1\n");
	printf("TEMOIN\n");
	printf("\n");
	printf("\n");

	//Paramètres
	txApprentissage = 0.25f;
	poids = 0.77f;
	MSEMin = 0.01f;

	//Création et Parametrage du réseau
	ann = fann_create_standard(3,5971,2985, 2);
	fann_randomize_weights(ann, -poids, poids);
	fann_set_learning_rate(ann, txApprentissage);

	//Apprentissage
	apprentissage(ann, learn_data, validation_data, MSEMin);

	//Verification des performances du réseau de neurones
	verification(ann, test_data);

	//Destruction du réseau
	fann_destroy(ann);

	printf("\n");
	printf("\n");


	///////////////////
	//TEST2////////////
	//TxApprentissage//
	///////////////////

	printf("////////////////////////////////////\n");
	printf("TEST2\n");
	printf("Augmentation du Taux d apprentissage\n");
	printf("\n");
	printf("\n");

	//Paramètres
	txApprentissage = 0.6f;
	poids = 0.77f;
	MSEMin = 0.01f;

	//Création et Parametrage du réseau
	ann = fann_create_standard(3,5971,2985, 2);
	fann_randomize_weights(ann, -poids, poids);
	fann_set_learning_rate(ann, txApprentissage);

	//Apprentissage
	apprentissage(ann, learn_data, validation_data, MSEMin);

	//Verification des performances du réseau de neurones
	verification(ann, test_data);

	//Destruction du réseau
	fann_destroy(ann);

	printf("\n");
	printf("\n");


	/////////
	//TEST3//
	//Poids//
	/////////

	printf("////////////////////\n");
	printf("TEST3\n");
	printf("Diminution des poids\n");
	printf("\n");
	printf("\n");

	//Paramètres
	txApprentissage = 0.25f;
	poids = 0.45f;
	MSEMin = 0.01f;

	//Création et Parametrage du réseau
	ann = fann_create_standard(3,5971,2985, 2);
	fann_randomize_weights(ann, -poids, poids);
	fann_set_learning_rate(ann, txApprentissage);

	//Apprentissage
	apprentissage(ann, learn_data, validation_data, MSEMin);

	//Verification des performances du réseau de neurones
	verification(ann, test_data);

	//Destruction du réseau
	fann_destroy(ann);
	
	printf("\n");
	printf("\n");


	//////////
	//TEST4///
	//MSEMin//
	//////////

	printf("/////////////////////////////\n");
	printf("TEST4\n");
	printf("Diminution du MSE à atteindre\n");
	printf("\n");
	printf("\n");

	//Paramètres
	txApprentissage = 0.25f;
	poids = 0.77f;
	MSEMin = 0.005f;

	//Création et Parametrage du réseau
	ann = fann_create_standard(3,5971,2985, 2);
	fann_randomize_weights(ann, -poids, poids);
	fann_set_learning_rate(ann, txApprentissage);

	//Apprentissage
	apprentissage(ann, learn_data, validation_data, MSEMin);

	//Verification des performances du réseau de neurones
	verification(ann, test_data);

	//Destruction du réseau
	fann_destroy(ann);

	printf("\n");
	printf("\n");


	return 0;
}


void apprentissage(struct fann* ann, struct fann_train_data* learn_data, struct fann_train_data* validation_data, float MSEMin){
	//Variables de calcul des erreurs
	float MSE = 1.f, MSE_learning = 0.f, MSE_validation = 1.f, MSE_tmp = 1.f;
	//Booléen
	int loop = 1, boolean = 0;
	int length = fann_length_train_data(validation_data);

	
	//Boucle principale d'apprentissage
	fann_save(ann,"saved-ann.net");		
	while(MSE_tmp > MSEMin && loop == 1)
	{
		float tmp = 0.f;
			
		for(int i=0; i<20; i++)
		{
			MSE_learning = fann_train_epoch(ann, learn_data);		
			printf("MSE du set d apprentissage %f\n", MSE_learning);
		}
		
		if (MSE_learning < MSE)
		{
			MSE = MSE_learning;

			for(int i=0; i<fann_length_train_data(validation_data); i++)
			{
				fann_type* output = fann_run(ann, validation_data->input[i]);
				tmp+= (*output - validation_data->output[i][0])*(*output - validation_data->output[i][0]);
			}
	
			MSE_tmp = tmp / length;
			printf("MSE du set de validation: %f\n", MSE_tmp);

			if (MSE_tmp <= MSE_validation)
			{
				fann_save(ann,"saved-ann.net");
				MSE_validation = MSE_tmp;
				boolean = 0;
			}

			else if(boolean == 1)
				loop = 0;
			
			else
				boolean++;
		}
	}
	printf("Fin de l apprentissage\n");
	printf("\n");
}


void verification(struct fann* ann, struct fann_train_data* test_data){

	int length = fann_length_train_data(test_data);

	//Variables de calcul de précision
	float accuracy=0.f , TPcount = 0.f, FPcount = 0.f, FNcount = 0.f, TNcount = 0.f;

	for(int i = 0; i<length; i++)
	{
		fann_type* output = fann_run(ann, test_data->input[i]);
			
		if(*output> 0.5 && test_data->output[i][0] > 0.5)
		{
			accuracy++;
			TPcount++;
		}
		else if(*output < 0.5 && test_data->output[i][0] > 0.5)
		{
			FNcount++;
		}
		else if(*output > 0.5 && test_data->output[i][0] < 0.5)
		{
			FPcount++;
		}
		else
		{
			accuracy++;
			TNcount++;
		}
	}
	accuracy /= length;
	TPcount /= length;
	TNcount /= length;
	FPcount /= length;
	FNcount /= length;

	//Affichage des résultats
	printf("Precision: %f\n", accuracy);
	printf("True Positif: %f\n", TPcount);
	printf("True Negatif: %f\n", TNcount);
	printf("False Positif: %f\n", FPcount);
	printf("False Negatif: %f\n", FNcount);
}