//Radial Basis Function neural network with SOFM (self-organizing feature map)
//written by Lei Jiang, Center for Computation and Technology, Louisiana State University

//the code is now for testbed only. The robustness of functionality is not guaranteed. The original author is glad to know any major modification of key functionality
//by the user of codes.

#include "read_datasets.h"

#define TAU 4
#define SOFM_THRESHOLD 0.1
#define WEIGHT_SOFM_DIMENSION    15 //6   //3   //10
#define STENCIL_THRESHOLD  99999   //999999    //2.5    //0.75    //0.6
#define MOMENTUM    0.05
#define RBF_LEARN_ITERATION 6000 //5000    //15000  //250

//added Dec6, 2009  for regularization
#define REGC 10.0

enum {DISCRETE =0, GAUSS = 1, MEXICAN_HAT = 2, FRENCH_HAT =3, SIMPLE =4};

//variables
double *hidden_RBF,
*output_RBF,
*target_RBF,
bias_RBF,
*weight_RBF,
**center_RBF,
*spread_RBF,
*errorsignal_hidden_RBF,
errorsignal_output_RBF,
*last_delta_weight;

double t_learn_rate;

//2D weight-vector arrays for SOFM
double **weight_SOFM;

//error
double currentEpsilon, epsilon = 0.0000001, min_distance_SOFM = 1000;
double *min_weight_SOFM;

int hidden_array_size_RBF,
max_patterns_RBF,
ytemp_RBF = 0,
ztemp_RBF = 0,
min_coord_x =0,
min_coord_y = 0,
currentIteration,
numofIteration = 2500;

//choose a kernel
int kernel = 1;  //1-gaussian 2-quadric 3-inverse quadric 4-linear 5-square 6-cubic 7-multiquadric
double beta= 50; //constant for some kernels


char filename_RBF[128];

double kernel_RBF(double x[], double center[], double spread, int size);
double output_function_RBF(double hidden[], double weight[], double bias, int size);

void Init_RBF(int pattern_count);
void learn_RBF();
void learn_SOFM();
int compare_output_to_target_RBF();
int compare_output_to_target_SOFM();
void validate_RBF();
void clear_memory_RBF();

//SOFM functions
double alpha(int t);
double t_l_rate(int t);
double sigma(int t);
double h(int x, int y, int t, int type);
double distance_2D(int a1, int a2, int b1, int b2);
double distance_nD(double x[], double w[], int n);
double find_min_coord(int num);
double weight_updating_SOFM(int x, int y, int t, int num);
void startEpoch_SOFM(int num);
void startLearning_SOFM();
void increment_pass_RBF(int pattern);

using namespace std;

void learn_RBF()
{
	int i,j,k, count =0;
	double pred[TRAINING_DATA_SIZE], obs[TRAINING_DATA_SIZE];
	double result = 0.0, temp_result = 0.0;
	number_of_input_patterns = TRAINING_DATA_SIZE;
	//learn_SOFM();
    
	cout << endl << "just press a key to continue to learn RBF" << endl;
	char temp_char = getch();
	if(temp_char == 'q' || temp_char == 'Q')
		return;

	//for(i =0; i< WEIGHT_SOFM_DIMENSION; i++)
	   //for(j =0; j< WEIGHT_SOFM_DIMENSION; j++)
	  /*for(i=0; i< CENTER_SIZE; i++)
	   {
         for(k =0; k< input_array_size; k++)
		   center_RBF[i][k] = input_center[i][k];
	   }*/
    init_center_RBF();

	//then calculate the sigma by computing distanced
    for(i =0; i< hidden_array_size_RBF; i++)
	{
		result = 0.0;
		count =0;
	     for(j =0; j< number_of_input_patterns; j++)
		 {
			 temp_result = 0.0;
			 for(k =0; k< input_array_size; k++)
				 temp_result+= (input[j][k] - center_RBF[i][k])*(input[j][k] - center_RBF[i][k]);
             temp_result/= input_array_size;
			 temp_result = sqrt(temp_result);

			 if(temp_result < STENCIL_THRESHOLD)
			 {
                 result += temp_result;
				 count++;
			 }
	
		 }
    	 spread_RBF[i] = sqrt(result/(double)count);
	}

	//then learn for the weight from hidden layer to output layer
	//while(!kbhit())
	for(k=0; k< RBF_LEARN_ITERATION; k++)
		//while(CE < 0.95)
	{
		if(kbhit())
			break;
    	for(i = 0; i< number_of_input_patterns; i++)
			increment_pass_RBF(i);

		if(compare_output_to_target_RBF()) {
			cout << endl << "iteration" << k <<":" <<endl<<endl;
			//break;
			//return;
		}
	}
		
	printf("Training stopped at iteration = %d \n", k);
	FILE *outfile;
 	strcpy(filename, LEARNING_FILE_PATH_RBF);
	outfile = fopen (filename, "w");
	if (!outfile)
	{
		printf ("Error opening %s\n", filename);
		exit(0);
	}
	for(i= 0; i< number_of_input_patterns; i++)
	{
		if(output_RBF[i] >= CLASS_THRESHOLD)
			pred[i] = 1.0;
		else
			pred[i] = 0.0;
		//pred[i] = (output_RBF[i]-outoffset)/outscale; 
		obs[i] = target[i];   //(target[i]-outoffset)/outscale;
	
		fprintf(outfile,"%f   %f   %f    %f\n",input[i][0], obs[i], pred[i], obs[i]-pred[i]);
	}
	calRMSE(pred,obs,number_of_input_patterns, RMSE);
	calMAE(pred,obs,number_of_input_patterns, MAE);
	//calCE(pred,obs,number_of_input_patterns, CE);
	calCD(pred,obs,number_of_input_patterns, CD);
	calRE(pred,obs,number_of_input_patterns, RE);
	//calAARE(pred,obs,number_of_input_patterns, AARE);
	fprintf(outfile,"training: root mean square error is %f \n", RMSE);
	fprintf(outfile,"training: mean absolute error is %f \n", MAE);
	//fprintf(outfile,"training: coefficient of efficiency is %f \n", CE);
	fprintf(outfile,"training: coefficient of determining is %f \n", CD);
	fprintf(outfile, "training: error rate is %f %%\n", RE*100);
	fclose(outfile);

	strcpy(filename, WEIGHT_I_H_FILE_RBF);
	outfile = fopen (filename, "w");
	if (!outfile)
	{
		printf ("Error opening %s\n", filename);
		exit(0);
	}
	for(i=0; i< hidden_array_size_RBF; i++)
	{
		for(j=0; j< input_array_size; j++)
			fprintf(outfile,"%f    ", center_RBF[i][j]);
		fprintf(outfile,"\n");
	}
	fclose(outfile);
	
	strcpy(filename, WEIGHT_H_O_FILE_RBF);
	outfile = fopen (filename, "w");
	if (!outfile)
	{
		printf ("Error opening %s\n", filename);
		exit(0);
	}
	for(i=0; i< hidden_array_size_RBF; i++)
	{
		fprintf(outfile,"%f", weight_RBF[i]);
		fprintf(outfile,"\n");
	}
	fclose(outfile);
	
	cout << endl << "learning successful" << endl;
	return;
}

void increment_pass_RBF(int pattern)
{
	int i;
	double mom = MOMENTUM;
    for(i=0; i< hidden_array_size_RBF; i++)
	{
		hidden_RBF[i] = kernel_RBF(input[pattern], center_RBF[i], spread_RBF[i], input_array_size);
	}

	output_RBF[pattern] = output_function_RBF(hidden_RBF, weight_RBF, bias_RBF, hidden_array_size_RBF);

	for(i =0; i< hidden_array_size_RBF; i++)
	{
		if(pattern >0)
			weight_RBF[i] += mom * last_delta_weight[i] + t_l_rate(pattern)* (target[pattern] - output_RBF[pattern]) * hidden_RBF[i];
			//editted Dec 6 for regularization
			//weight_RBF[i] += mom * last_delta_weight[i] + t_l_rate(pattern)* ((target[pattern] - output_RBF[pattern])+REGC/2*weight_RBF[i]*weight_RBF[i])* hidden_RBF[i];
		else
			weight_RBF[i] +=  t_l_rate(pattern)* (target[pattern] - output_RBF[pattern]) * hidden_RBF[i];
			//editted Dec 6 for regularization
		    //weight_RBF[i] +=  t_l_rate(pattern)*((target[pattern] - output_RBF[pattern])+REGC/2*weight_RBF[i]*weight_RBF[i])* hidden_RBF[i];
		last_delta_weight[i] = mom * last_delta_weight[i] + t_l_rate(pattern)* (target[pattern] - output_RBF[pattern]) * hidden_RBF[i];
		//editted for regularization
		//last_delta_weight[i] = mom * last_delta_weight[i] + t_l_rate(pattern)* ((target[pattern] - output_RBF[pattern])+REGC/2*weight_RBF[i]*weight_RBF[i]) * hidden_RBF[i];
	}

	if(pattern >0)
		//bias_RBF += mom * last_delta_weight[i] + t_l_rate(pattern)* (target[pattern] - output_RBF[pattern]);
		bias_RBF = 0.0;   //for classification problems Jan 23
		//bias_RBF += mom * last_delta_weight[i] + t_l_rate(pattern)* ((target[pattern] - output_RBF[pattern])+REGC/2*weight_RBF[i]*weight_RBF[i]);
	else
		//bias_RBF +=  t_l_rate(pattern)* (target[pattern] - output_RBF[pattern]);
		bias_RBF = 0.0;
		//bias_RBF +=  t_l_rate(pattern)* ((target[pattern] - output_RBF[pattern])+REGC/2*weight_RBF[i]*weight_RBF[i]);
	last_delta_weight[i] = mom * last_delta_weight[i] + t_l_rate(pattern)* (target[pattern] - output_RBF[pattern]);
	//last_delta_weight[i] = mom * last_delta_weight[i] + t_l_rate(pattern)* ((target[pattern] - output_RBF[pattern])+REGC/2*weight_RBF[i]*weight_RBF[i]);
}

void learn_SOFM()
{
    int	i,j,k,x = rand();
	srand(x);
    for(i =0; i< WEIGHT_SOFM_DIMENSION; i++)
		for(j=0; j< WEIGHT_SOFM_DIMENSION; j++)
			for(k=0; k< input_array_size; k++)
				weight_SOFM[i*WEIGHT_SOFM_DIMENSION+j][k] = ((double)rand()/(double)RAND_MAX)/500.00 - 0.001;

	if (file_loaded == 0)
	{
		cout << endl
			<< "there is no data loaded into memory"
			<< endl;
		return;
	}
	cout << endl << "learning SOFM..." << endl;
	register int y;
	//while(!kbhit())
		//for(k=0; k<200; k++)
		//while(CE < 0.95)
	//{
		int iterations = 0;
		while (iterations<= numofIteration && currentEpsilon > epsilon)
		{
			
			for(y= PRED_PERIOD; y<number_of_input_patterns; y++) 
			{
				startEpoch_SOFM(y);
			}
			iterations++;
			
			if(compare_output_to_target_SOFM())
			{
				cout<< "learning SOFM successful ..."<<endl;
				break;
			}
		}
	//}
 	cout<< "learning SOFM successful ..."<<endl;
	return;
}

int compare_output_to_target_RBF()
{
	register int y,count=0;
	register double temp;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	double pred[TRAINING_DATA_SIZE], obs[TRAINING_DATA_SIZE];
	
	for(y= 0 ; y < number_of_input_patterns; y++) {
		//pred[y] = (output_RBF[y]-outoffset)/outscale; 
		if(output_RBF[y] >= CLASS_THRESHOLD)
			pred[y] = 1.0;
		else
			pred[y] = 0.0;
		obs[y] =  target[y];   //(target[y]-outoffset)/outscale; 
		
		temp = obs[y] - pred[y];
		if(temp == 0)
            count++;
	}
	/*calRMSE(pred,obs,number_of_input_patterns, RMSE);
	calMAE(pred,obs,number_of_input_patterns, MAE);
	//calCE(pred,obs,number_of_input_patterns, CE);
	calCD(pred,obs,number_of_input_patterns, CD);
	calRE(pred,obs,number_of_input_patterns, RE);
	//calAARE(pred,obs,number_of_input_patterns, AARE);
	printf("training: root mean square error is %f \n", RMSE);
	printf("training: mean absolute error is %f \n", MAE);
	//printf("training: coefficient of efficiency is %f \n", CE);
	printf("training: coefficient of determining is %f \n", CD);
	printf("training: error rate is %f %% \n", RE*100);*/
	printf("training: error rate is %f %% \n", (double)count/(double)number_of_input_patterns*100);
	printf("_____________________________________\n");
	
	if(CE < maxCE)
		return 1;
	else
	{
		maxCE = CE;
		return 0;
	}
}

int compare_output_to_target_SOFM()
{
	 int i,j,k;
     double dist[WEIGHT_SOFM_DIMENSION][WEIGHT_SOFM_DIMENSION];
	 double temp = 0.0, temp_min = 1000.00;

	 for(i=0; i< WEIGHT_SOFM_DIMENSION; i++)
		 for(j=0; j<WEIGHT_SOFM_DIMENSION; j++)
			 dist[i][j] = 0.0;

     for(j=0; j< WEIGHT_SOFM_DIMENSION; j++)
		 for(k=0; k< WEIGHT_SOFM_DIMENSION; k++)
		 {
			 for(i = PRED_PERIOD; i< number_of_input_patterns; i++)
			 {
				dist[j][k] += distance_nD(input[i],weight_SOFM[j*WEIGHT_SOFM_DIMENSION+k], input_array_size);
			 }
			 dist[j][k] /= (number_of_input_patterns-PRED_PERIOD);
		 }
	 /*for(j=0; j< WEIGHT_SOFM_DIMENSION; j++)
	 {
		 for(k=0; k< WEIGHT_SOFM_DIMENSION; k++)
		 {
			 printf("%f    ", dist[j][k]);
		 }
		 printf("\n");
	 }*/
	 for(j=0; j< WEIGHT_SOFM_DIMENSION; j++)
	 {
		 for(k=0; k< WEIGHT_SOFM_DIMENSION; k++)
		 {
			temp += dist[j][k];
		 }
	 }
	 temp /= (WEIGHT_SOFM_DIMENSION*WEIGHT_SOFM_DIMENSION);
	 printf("Weight Dimension: %d        the average SOFM distance is %f \n", WEIGHT_SOFM_DIMENSION, temp);
     /*if(min_distance_SOFM > dist)
	 {
		 min_distance_SOFM = dist;
		 for(i=0; i< input_array_size; i++)
			 min_weight_SOFM[i] = weight_SOFM[min_coord_x][min_coord_y][i];
		
		 return 0;
	 }
	 else
	 {
		 for(i=0; i< input_array_size; i++)
			 printf("SOFM center %d: %f \n", i, min_weight_SOFM[i]);
		 return 1;
	 }*/
	 return 0;
}

void clear_memory_RBF()
{
	int x;
	for(x=0; x<number_of_input_patterns; x++)
	{
		delete [] input[x];
	}
	delete [] input;
	delete [] hidden_RBF;
	delete [] output_RBF;

	delete [] target;
	delete [] target_RBF;
	delete [] weight_RBF;
	for(x=0; x< hidden_array_size_RBF; x++)
	{
		delete [] center_RBF[x];
	}
	delete [] center_RBF;
	delete [] spread_RBF;

	for(x=0; x< WEIGHT_SOFM_DIMENSION*WEIGHT_SOFM_DIMENSION; x++)
	{
		delete [] weight_SOFM[x];
	}
	delete [] weight_SOFM;
	delete [] center_RBF;
	delete [] min_weight_SOFM;
	delete [] last_delta_weight;
	delete [] errorsignal_hidden_RBF;
	file_loaded = 0;
	return;
}

void Init_RBF(int pattern_count)
{
	int i, j, x=1;
	number_of_input_patterns = TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE;
	basic_memory_assign();
	hidden_array_size_RBF = CENTER_SIZE;  //WEIGHT_SOFM_DIMENSION*WEIGHT_SOFM_DIMENSION

	hidden_RBF = new double [hidden_array_size_RBF];
	if(!hidden_RBF) { cout << endl << "memory problem!"; exit(1); }
	output_RBF = new double [number_of_input_patterns];
	if(!output_RBF) { cout << endl << "memory problem!"; exit(1); }

	weight_RBF = new double [hidden_array_size_RBF];
	if(!weight_RBF) { cout << endl << "memory problem!"; exit(1); }
	center_RBF = new double * [hidden_array_size_RBF];
	for(i=0; i< hidden_array_size_RBF; i++)
	{
		center_RBF[i] = new double [IN_ARRAY_SIZE];
		if(!center_RBF[i]) { cout << endl << "memory problem!"; exit(1); }
	}
	spread_RBF = new double [hidden_array_size_RBF];
	if(!spread_RBF) { cout << endl << "memory problem!"; exit(1); }

	errorsignal_hidden_RBF = new double [hidden_array_size_RBF];
	if(!errorsignal_hidden_RBF) { cout << endl << "memory problem!"; exit(1); }
	errorsignal_output_RBF = 0.0;

	weight_SOFM = new double* [WEIGHT_SOFM_DIMENSION*WEIGHT_SOFM_DIMENSION];
	for(i=0; i< WEIGHT_SOFM_DIMENSION*WEIGHT_SOFM_DIMENSION; i++)
	{
		weight_SOFM[i] = new double [hidden_array_size_RBF];
		if(!weight_SOFM[i]) { cout << endl << "memory problem!"; exit(1); }
	}

	min_weight_SOFM = new double [input_array_size];
	if(!min_weight_SOFM) { cout << endl << "memory problem!"; exit(1); }

	last_delta_weight = new double [hidden_array_size_RBF+1];
	if(!last_delta_weight) { cout << endl << "memory problem!"; exit(1); }

	for(i=0; i< hidden_array_size_RBF; i++)
	{
		srand(x);
		weight_RBF[i] = (double)rand()/2000000.0;   //(double)RAND_MAX/50000.0;
			x++;
		for(j =0; j< input_array_size; j++)
		     center_RBF[i][j] = 0.0;
		spread_RBF[i] = 0.0;
	}

	for(i=0; i< hidden_array_size_RBF+1; i++)
		last_delta_weight[i] = 0.0;
   	bias_RBF = (double)rand()/(double)RAND_MAX/500.00 - 0.001;
	//bias_RBF = (double)rand()/50000;
	currentIteration = 0;
	currentEpsilon = 1000;

	init_input_output();
	init_cweights();
	return;

}

void validate_RBF()
{
	register int x,y,i,j,k, count;
	register double temp,temp_result, result;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	CE = 0.0;
	AARE = 0.0;
	double pred[VALIDATION_DATA_SIZE], obs[VALIDATION_DATA_SIZE];

//    read_naive_tf_file2(TRAINING_TRAFFIC_FILE_PATH2, 1, TRAINING_DATA_SIZE);
	read_cos_file("cos_rand.dat",TRAINING_DATA_SIZE+1,TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE);
	//read_max_wave_file("maxwave_oct20_rand.dat",1,VALIDATION_DATA_SIZE);
	if(input)
	  delete []input;
	if(output_RBF)
	  delete []output_RBF;
	if(hidden_RBF)
	  delete []hidden_RBF;
	if(spread_RBF)
	   delete []spread_RBF;
	if(target)
	   delete []target;

	number_of_input_patterns = VALIDATION_DATA_SIZE;
	input = new double * [number_of_input_patterns];
	if(!input) { cout << endl << "memory problem!"; exit(1); }
	for(x=0; x<number_of_input_patterns; x++)
	{
		input[x] = new double [input_array_size];
		if(!input[x]) { cout << endl << "memory problem!"; exit(1); }
	}
	hidden_RBF = new double [hidden_array_size_RBF];
	if(!hidden_RBF) { cout << endl << "memory problem!"; exit(1); }
	spread_RBF = new double [hidden_array_size_RBF];
	if(!spread_RBF) { cout << endl << "memory problem!"; exit(1); }
	output_RBF = new double [number_of_input_patterns];
	if(!output_RBF) { cout << endl << "memory problem!"; exit(1); }

	target = new double [number_of_input_patterns];
	if(!target) { cout << endl << "memory problem!"; exit(1); }

	wave_scale = 0.58;
	init_cweights();
	init_input_output(); 

	for(i =0; i< hidden_array_size_RBF; i++)
	{
		result = 0.0;
		count =0;
		for(j =0; j< number_of_input_patterns; j++)
		{
			temp_result = 0.0;
			for(k =0; k< input_array_size; k++)
				temp_result+= (input[j][k] - center_RBF[i][k])*(input[j][k] - center_RBF[i][k]);
			temp_result/= input_array_size;
			temp_result = sqrt(temp_result);

			if(temp_result < STENCIL_THRESHOLD)
			{
				result += temp_result;
				count++;
			}

		}
		spread_RBF[i] = sqrt(result/(double)count);
	}

    for(pattern =0; pattern< number_of_input_patterns; pattern++)
	{
		for(x=0; x< hidden_array_size_RBF; x++)
		{
		   hidden_RBF[x] = kernel_RBF(input[pattern], center_RBF[x], spread_RBF[x], input_array_size);
		}
		output_RBF[pattern] = output_function_RBF(hidden_RBF, weight_RBF, bias_RBF, hidden_array_size_RBF);
	}
	FILE *outfile1;
	strcpy(filename, VALIDATION_OUT_FILE_PATH_RBF);
	outfile1 = fopen (filename, "w");
	if (!outfile1)
	{
    	printf ("Error opening %s\n", filename);
		exit(0);
	}
    for(y=0; y < number_of_input_patterns; y++) {
			pred[y] = output_RBF[y]; //rev_sigmoid(1.25*(output_RBF[y]-0.1))*wave_scale+15;
		    obs[y] = target[y]; //rev_sigmoid(1.25*(target[y]-0.1))*wave_scale+15;
					
			temp = obs[y] - pred[y];

			fprintf(outfile1,"%f   %f    %f\n",obs[y], pred[y], temp);
	}
	calRMSE(pred,obs,number_of_input_patterns, RMSE);
	calMAE(pred,obs,number_of_input_patterns, MAE);
	//calCE(pred,obs,number_of_input_patterns, CE);
	calCD(pred,obs,number_of_input_patterns, CD);
	calRE(pred,obs,number_of_input_patterns, RE);
	//calAARE(pred,obs,number_of_input_patterns, AARE);

	printf("validation: root mean square error is %f \n", RMSE);
	printf("validation: mean absolute error is %f \n", MAE);
	//printf("validation: coefficient of efficiency is %f \n", CE);
	printf("validation: coefficient of determining is %f \n", CD);
	printf("validation: error rate is %f %% \n", RE*100);
	fprintf(outfile1,"validation: root mean square error is %f \n", RMSE);
	fprintf(outfile1,"validation: mean absolute error is %f \n", MAE);
	//fprintf(outfile1,"validation: coefficient of efficiency is %f \n", CE);
	fprintf(outfile1,"validation: coefficient of determining is %f \n", CD);
	fprintf(outfile1, "validation: error rate is %f  %%\n", RE*100);
	fclose(outfile1);
	return;
}

//subroutines for Radial basis function
double kernel_RBF(double x[], double center[], double spread, int size)
{
	int i;
	double sum = 0.0;
    for(i=0; i< size; i++)
		sum+=(x[i]-center[i])*(x[i]-center[i]);
	sum /= (-2*spread*spread);
	switch(kernel)
	{
	case 1:  //Gaussian kernel
	   return exp(sum);  
	case 2: //quadric kernel
		return sqrt(sum*sum+beta*beta);
	case 3:    //inverse quadric kernel
		return 1/sqrt(sum*sum+beta*beta);
	case 4:  //linear kernel
		return sum;
	case 5:
		return sum*sum;
	case 6:
		return sum*sum*sum;
	case 7:
		return sqrt(beta*beta*sum*sum+1);
	default:
		return sqrt(sum*sum+beta*beta);
	}
}

double output_function_RBF(double hidden[], double weight[], double bias, int size)
{
	int i;
    double output = 0.0;
	output+=bias;
	
	for(i=0; i<size; i++)
		if(hidden[i] >= (double)pow((double)10.0,-9))
		    output+= hidden[i] *weight[i];
	output = sigmoid(output);
	return output;
}

//SOFM functions
double alpha(int t)
{
	return 0.1*exp((double)(-t)/(double)TAU);
}

double t_l_rate(int t)
{
	return 0.05*(1.0 - (double)t/(double)(number_of_input_patterns));    //-PRED_PERIOD));
}

double sigma(int t)
{
	return exp((double)t/(double)TAU);
}

double distance_2D(int a1, int a2, int b1, int b2)
{
	return (double)sqrt((double)((a1-b1)*(a1-b1)+(a2-b2)*(a2-b2)));
}

double distance_nD(double x[], double w[], int n)
{
	int i;
	double sum = 0.0;
	for(i=0; i< n; i++)
		sum+= (x[i]-w[i])*(x[i]-w[i]);

	return (double)sqrt(sum);
}

double h(int x, int y, int t, int type)
{
	double result = 0.0;
	double distance = distance_2D(x,y,min_coord_x,min_coord_y);
	double s = sigma(t);
	int a =2;
	switch(type)
	{
	case 0:
		distance = abs(x-min_coord_x) + abs(y-min_coord_y);
	    result = exp((double)(-distance));
		break;
	case 1:
		result = exp(-distance*distance/(s*s));
		break;
	case 2:
		result = exp(-distance*distance/(s*s))*(1.0-2.0*distance*distance/(s*s));
		break;
	case 3:
		if(distance <= a)
			result = 1;
		else if (distance >a && distance <= 3*a)
			result = -1.0/3.0;
		else if (distance > 3*a)
			result = 0;
		break;
	case 4:
		if(distance >= 2*a)
			result =0;
		else
			result =1;
		break;
	default:
		result = exp(-distance*distance/(s*s));
	}

	return result;
}

double find_min_coord(int num)
{
	int i,j;
	double dist = 0.0, min_dist = 1000.00;
	for(i=0; i< WEIGHT_SOFM_DIMENSION; i++)
		for(j=0; j< WEIGHT_SOFM_DIMENSION; j++)
		{
             dist = distance_nD(input[num], weight_SOFM[i*WEIGHT_SOFM_DIMENSION+j], input_array_size);
			 if(dist < min_dist)
			 {
				 min_dist = dist;
				 min_coord_x = i;
				 min_coord_y = j;
			 }
		}
	return min_dist;	
}

double weight_updating_SOFM(int x, int y, int t, int num)  //num is the order number of input pattern
{  
	double avgDelta = 0;
    double modificationValue =0;
	int i;
    for (i = 0; i < input_array_size; i++)
	{
		//modificationValue = alpha(t) * h(x,y,t,SIMPLE) * (input[num][i] - weight_SOFM[x][y][i]);
		modificationValue = alpha(t) * h(x,y,t,MEXICAN_HAT) * (input[num][i] - weight_SOFM[x*WEIGHT_SOFM_DIMENSION+y][i]);
		weight_SOFM[x*WEIGHT_SOFM_DIMENSION+y][i] += (float)modificationValue;
		avgDelta += modificationValue;
	}
	avgDelta = avgDelta / input_array_size;
	return avgDelta;
}

void startEpoch_SOFM(int num)
{
	int i,j;
	double min = find_min_coord(num);
	if(min < SOFM_THRESHOLD)
        return;

	currentEpsilon = 0;
	for (i = 0; i < WEIGHT_SOFM_DIMENSION; i++)
		for (j = 0; j < WEIGHT_SOFM_DIMENSION; j++)
		{
			currentEpsilon += weight_updating_SOFM(i,j, currentIteration, num);                   
		}
		currentIteration++;
	currentEpsilon = fabs((currentEpsilon/(double)(WEIGHT_SOFM_DIMENSION * WEIGHT_SOFM_DIMENSION)));
}

void startLearning_SOFM()
{
	int iterations = 0, i;
	while (iterations<= numofIteration && currentEpsilon > epsilon)
	{
		
		for (i =  PRED_PERIOD; i < number_of_input_patterns; i++)
		{
			startEpoch_SOFM(i);
		}
		iterations++;
	}
}

void read_and_go_RBF(char* infile1, char* infile2)
{
	int i=0, j=0, k=0; 
	int count =0; //count the num of edges
	int firstmark =0;
	char tempstr[MAX_LINE_LENGTH];
	char  tempchar[LINE_THRESHOLD];
	for(i=0;i< MAX_LINE_LENGTH;i++)
		tempstr[i] = '\0';
	
	FILE *file;
	file = fopen (infile1, "r");
	if (!file)
	{
		printf ("Error opening %s\n", infile1);
		exit(0);
	}
	
	while(fgets(tempstr,MAX_LINE_LENGTH,file))
	{
		i = 0;
		firstmark = 0;
		
		for(j=0; j< LINE_THRESHOLD; j++)
			tempchar[j] = '\0';
		
		if(tempstr[0] == '"' || tempstr[0] == '#')
			continue;
		
		for(k=0; k< input_array_size; k++)
		{
			while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
			{
				tempchar[firstmark] = tempstr[i];
				i++;
				firstmark++;
			}
			
			center_RBF[count][k] = atof(tempchar);
			
			for(j=0; j< firstmark; j++)
				tempchar[j] = '\0';
			firstmark = 0;

			while(tempstr[i] == ' ' || tempstr[i] == 9)
			{
				i++;
			}
			
		}
		count++;
		
	}
	
    fclose(file);

	count =0;
	file = fopen (infile2, "r");
	if (!file)
	{
		printf ("Error opening %s\n", infile2);
		exit(0);
	}
	
	while(fgets(tempstr,MAX_LINE_LENGTH,file))
	{
		i = 0;
		firstmark = 0;
		
		for(j=0; j< LINE_THRESHOLD; j++)
			tempchar[j] = '\0';
		
		if(tempstr[0] == '"' || tempstr[0] == '#')
			continue;
		
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
			
		weight_RBF[count] = atof(tempchar);
	
		count++;
	}
	
    fclose(file);
}




