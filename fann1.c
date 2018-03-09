#include <stdio.h>
#include <stdlib.h>
#include <fann.h>

/*
Nb neurones cachés
Tx apprentissage
Poids
MSE min
*/

int main() 
{
	struct fann* ann = fann_create_standard(3,5971,2985, 2);
	struct fann_train_data *train_data, *test_data, *learn_data, *validation_data;
	int inputVect = 0, length = 0, i, j;

	//Variables de calcul de précision
	float accuracy=0.f , TPcount = 0.f, FPcount = 0.f, FNcount = 0.f, TNcount = 0.f;
	//Variables de calcul des erreurs
	float MSE = 1.f, MSE_learning = 0.f, MSE_validation = 1.f, MSE_tmp = 0.f;
	//Booléen
	int loop = 1;

	//Parametrage du réseau
	fann_randomize_weights(ann, -0.77, 0.77);
	fann_set_learning_rate(ann, 0.25);

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

	length = fann_length_train_data(validation_data);

	
	//Boucle principale d'apprentissage
	fann_save(ann,"saved-ann.net");		
	while(MSE > 0.01f && loop == 1)
	{
		float tmp = 0.f;
			
		for(i=0; i<10; i++)
		{
			MSE_learning = fann_train_epoch(ann, learn_data);		
			printf("MSE_learning: %f\n", MSE_learning);
		}
		
		if (MSE_learning < MSE)
		{
			MSE = MSE_learning;

			for(i=0; i<fann_length_train_data(validation_data); i++)
			{
				fann_type* output = fann_run(ann, validation_data->input[i]);
				tmp+= (*output - validation_data->output[i][0])*(*output - validation_data->output[i][0]);
			}	
	
			MSE_tmp = tmp / length;
			printf("MSE_tmp: %f\n", MSE_tmp);

			if (MSE_tmp <= MSE_validation){
				fann_save(ann,"saved-ann.net");
				MSE_validation = MSE_tmp;
			}

			else
				loop = 0;

		}
		
	}

	//Verification des performances du réseau de neurones
	length = fann_length_train_data(test_data);
	for(i=0; i<fann_length_train_data(test_data); i++)
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
	printf("acc: %f\n", accuracy);
	printf("TP: %f\n", TPcount);
	printf("TN: %f\n", TNcount);
	printf("FP: %f\n", FPcount);
	printf("FN: %f\n", FNcount);

	//Destruction du réseau
	fann_destroy(ann);
	
	return 0;
}
