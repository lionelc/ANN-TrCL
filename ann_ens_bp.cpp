//Backpropagation artificial neural network
//modified by Lei Jiang, Center for Computation and Technology, Louisiana State University

//the code is now for testbed only. The robustness of functionality is not guaranteed. The original author is glad to know any major modification of key functionality
//by the user of codes.

#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <float.h>
#include <string.h>

#include "ann_ens_bp.h"
#include "learn_RBF_SOFM.h"
#include "tradaboost.h"
#include "trbagg.h"
#include "trcl.h"

#include "corrtest.h"   //test of correlation

using namespace std;

int main()
{
	for(;;) {
		char choice;
		char choice_struct;
		cout << "Please input the structure of ANN: " << endl;
		cout << "1. Multi-layer Perceptron;  2. Radial basis function" <<endl;
		do { 
			choice_struct = getch(); 
			if(choice_struct == 'q' || choice_struct == 'Q')
				return 0;
		} 
		while (choice_struct != '1' && choice_struct != '2');
		/*cout << "Please input the learning mode: " << endl;
		cout << "1. Batch mode; 2. Incremental mode" << endl;
				do { 
			         choice_lmode = getch(); 
			         if(choice_lmode == 'q' || choice_lmode == 'Q')
				    return 0;
		           } 
		while (choice_lmode != '1' && choice_lmode != '2');*/
		cout << endl << "1. load training data" << endl;
		cout << "2. learn from data" << endl;
		cout << "3. read weights from learned cases" << endl;
		cout << "4. quality control" << endl;
		cout << "5. change learning rate" << endl;
		cout << "6. validation" << endl ;
		//added by LJ
		cout << "7. exit" << endl;
		cout << "a. TrAdaBoost" << endl;
		cout << "b. TrBagg" << endl;
		cout << "c. TrCL" << endl;
		cout << "d. correlation test" <<endl << endl;
		cout << "Enter your choice (1-8, a,b,c)"<<endl;
		do { choice = getch(); } while (choice != '1' && choice != '2' && choice != '3' && choice != '4' && choice != '5' && choice != '6' && choice!= '7' && choice!='a' && choice!='b' && choice!='c' && choice!='d');
		switch(choice) {
			case '1':
			{
				if (file_loaded == 1) 
				{
					if(choice_struct == '1')
						clear_memory();
					else if(choice_struct == '2')
						clear_memory_RBF();
				}
				file_loaded = 1;
				init_data_memory();
				//read_letter_file("letter-recognition.data",1,TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE);
				//read_mushroom_file("agaricus-lepiota.data",1,TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE);
				read_mushroom_file("mushroom_transfer-new2.data",1,TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE); //mushroom_transfer-new2.data
				//read_aus_file("australian-transfer2.data", 1, TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE);
			    if(choice_struct == '1')
		     	    Initialization(TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE);
				else if(choice_struct == '2')
				    Init_RBF(TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE);
			}
			break;
			case '2':
			{
			    if(choice_struct == '1')
					learn();
				else if(choice_struct == '2')
					learn_RBF();
			}
			break;
			case '3': 
			////////////////////////////read weights from learning file and perform learning
			{
			   	if(choice_struct == '1')
					read_and_go(WEIGHT_I_H_FILE,WEIGHT_H_O_FILE);
				else if(choice_struct == '2')
					read_and_go_RBF(WEIGHT_I_H_FILE_RBF,WEIGHT_H_O_FILE_RBF);
			}
			break;
			case '4':
			{
			   //leave blank intentionally
			}
			break;
			case '5': change_learning_rate();
			break;
			case '6':
			{
				if(choice_struct == '1')
					validate();
				else if(choice_struct == '2')
				    validate_RBF();
			}
		    break;
			case '7': return 0;
			break;
			case 'a':
				tradaboost();
			break;
			case 'b':
				trbagg();
			break;
			case 'c':
				trcl();
			break;
			case 'd':
				corrtest();

		};
	}
}

//added Jul 20
void basic_memory_assign()
{
	int x;
	input = new double * [number_of_input_patterns];
	if(!input) { cout << endl << "memory problem!"; exit(1); }
	for(x=0; x<number_of_input_patterns; x++)
	{
		input[x] = new double [input_array_size];
		if(!input[x]) { cout << endl << "memory problem!"; exit(1); }
	}

	target = new double [number_of_input_patterns];
	if(!target) { cout << endl << "memory problem!"; exit(1); }
	input_array_size = IN_ARRAY_SIZE;  //7;    //4    //8;
	learning_rate = LEARN_RATE;  //original 0.5
}

//added Jan 23 for data reading
void init_data_memory()
{
    input_mushroom = new int * [TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE];
	//input_aus = new double * [TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE];
	if(!input_mushroom) { cout << endl << "memory problem!"; exit(1); }
	//if(!input_aus) { cout << endl << "memory problem!"; exit(1); }
	for(int x=0; x< TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE; x++)
	{
		input_mushroom[x] = new int [IN_ARRAY_SIZE];
		if(!input_mushroom[x]) { cout << endl << "memory problem!"; exit(1); }
		/*input_aus[x] = new double [IN_ARRAY_SIZE];
		if(!input_aus[x]) { cout << endl << "memory problem!"; exit(1); }*/
	}
	mushroom_class = new int [TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE];
	//aus_class = new int [TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE];
	return;
}

void initialize_net()
{
	int x,y;
	hidden = new double *[NUM_ENS+3];
	if(!hidden) { cout << endl << "memory problem!"; exit(1); }
	for(x=0; x< NUM_ENS+3; x++)
	{
	    hidden[x] = new double [hidden_array_size];
		if(!hidden[x]) { cout << endl << "memory problem!"; exit(1); }
	}
	
	output = new double *[NUM_ENS+3];
	if(!output) { cout << endl << "memory problem!"; exit(1); }
	for(x=0; x< NUM_ENS+3;x++)
	{
	   output[x] = new double [TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE];
	   if(!output[x]) { cout << endl << "memory problem!"; exit(1); }
	}
	
    bias = new double *[NUM_ENS+3];
	if(!bias) { cout << endl << "memory problem!"; exit(1); }
	for(x=0; x< NUM_ENS+3;x++)
	{
	  bias[x] = new double [bias_array_size];
	  if(!bias[x]) { cout << endl << "memory problem!"; exit(1); }
	}
	
    weight_i_h = new double ** [NUM_ENS+3];
	if(!weight_i_h) { cout << endl << "memory problem!"; exit(1); }
	for(x=0; x< NUM_ENS+3;x++)
	{
	   weight_i_h[x] = new double * [input_array_size];
	   if(!weight_i_h[x]) { cout << endl << "memory problem!"; exit(1); }
	}
	for(x=0; x< NUM_ENS+3;x++)
	  for(y=0; y<input_array_size; y++)
	{
		weight_i_h[x][y] = new double [hidden_array_size];
		if(!weight_i_h[x][y]) { cout << endl << "memory problem!"; exit(1); }
	}

    weight_h_o = new double * [NUM_ENS+3];
    if(!weight_h_o) { cout << endl << "memory problem!"; exit(1); }
	for(x=0; x< NUM_ENS+3;x++)
	{
	   weight_h_o[x] = new double [hidden_array_size];
	   if(!weight_h_o[x]) { cout << endl << "memory problem!"; exit(1); }
	}	
	
	errorsignal_hidden = new double *[NUM_ENS+3];
    if(!errorsignal_hidden) { cout << endl << "memory problem!"; exit(1); }
	for(x=0; x< NUM_ENS+3; x++)
	{
	   errorsignal_hidden[x] = new double [hidden_array_size];
	   if(!errorsignal_hidden[x]) { cout << endl << "memory problem!"; exit(1); }
	}

	errorsignal_output = new double [NUM_ENS+3];
	if(!errorsignal_output) { cout << endl << "memory problem!"; exit(1); }
   	
    if(choice_lmode == '1')
	{
        delta_bias = new double *[NUM_ENS+3];
		if(!delta_bias) { cout << endl << "memory problem!"; exit(1); }
		for(x=0; x< NUM_ENS+3; x++)
		{
		   delta_bias[x] = new double [bias_array_size];
		   if(!delta_bias[x]) { cout << endl << "memory problem!"; exit(1); }
		}
		delta_weight_i_h = new double ** [NUM_ENS+3];
		if(!delta_weight_i_h) { cout << endl << "memory problem!"; exit(1); }
		for(x=0; x< NUM_ENS+3; x++)
		{
		   delta_weight_i_h[x] = new double * [input_array_size];
		   if(!delta_weight_i_h[x]) { cout << endl << "memory problem!"; exit(1); }
		}
		for(x=0; x< NUM_ENS+3; x++)
		   for(y=0; y<input_array_size; y++)
		{
			delta_weight_i_h[x][y] = new double [hidden_array_size];
			if(!delta_weight_i_h[x][y]) { cout << endl << "memory problem!"; exit(1); }
		}

		delta_weight_h_o = new double * [NUM_ENS+3];
		if(!delta_weight_h_o) { cout << endl << "memory problem!"; exit(1); }
		for(x=0; x< NUM_ENS+3; x++)
		{
		   delta_weight_h_o[x] = new double  [hidden_array_size];
		   if(!delta_weight_h_o[x]) { cout << endl << "memory problem!"; exit(1); }
		}
	}
	return;
}

void learn()
{
	int i,j,count;
	iter = 0;
	number_of_input_patterns = TRAINING_DATA_SIZE;
	double pred[TRAINING_DATA_SIZE], obs[TRAINING_DATA_SIZE],accuracy;
	if (file_loaded == 0)
	{
		cout << endl
			 << "there is no data loaded into memory"
			 << endl;
		return;
	}
	cout << endl << "learning..." << endl << "press a key to return to menu" << endl;
	register int y, temp,x;

	//FILE *learn_file;
	//strcpy(filename, LEARNING_CURVE_PATH);
	//learn_file = fopen (filename, "w");
	while(!kbhit())
	{
		if(choice_lmode == '2')
		{
			hidden_max= output_max = 0;
			hidden_min = output_min = 9999;
		      for(y=0; y<number_of_input_patterns; y++) {
				    for(i=0; i< ensn; i++)
				       ens_forward_pass(i,y);
				   for(i=0; i< ensn; i++)
			         ens_backward_pass(i,y,NCL_LAMBDA);
		    }
			hidden_range = hidden_max - hidden_min;
			output_range = output_max - output_min;
		}
		else if(choice_lmode == '1')
		{
			hidden_max= output_max = 0;
			hidden_min = output_min = 9999;

			for(i=0; i< ensn; i++)
			{
             for(y=0; y<number_of_input_patterns; y++) {
			   ens_forward_pass(i,y);
			   ens_backward_pass_batch(i,y);
		    }
            hidden_range = hidden_max - hidden_min;
			output_range = output_max - output_min;
			for(x=0; x<hidden_array_size; x++) {
		       		weight_h_o[i][x]+=delta_weight_h_o[i][x];
					delta_weight_h_o[i][x] = 0;
			}
			for(x=0; x<input_array_size; x++) {
		        for(y=0; y<hidden_array_size; y++) {
				    weight_i_h[i][x][y]+=delta_weight_i_h[i][x][y];
					delta_weight_i_h[i][x][y] =0;
		        }
	         }
	        for(x=0; x<hidden_array_size; x++) {
	        		bias[i][x]+=delta_bias[i][x];
					delta_bias[i][x] = 0;
	         }
			}
		}
	    iter++;
		/*if (!learn_file)
		{
			printf ("Error opening %s\n", filename);
			exit(0);
		}*/

		if(compare_output_to_target()) {
			cout << endl << "training finished" << endl;
			cout << endl << "just press a key to continue to write learning error into file" << endl;
			char temp_char = getch();
			break;
		}
		//else
			//fprintf(learn_file,"%d    %f   %f   %f   %f\n",(int)iter, RMSE, MAE, CE, CD);
	}
	//fclose(learn_file);

    FILE *outfile;
	strcpy(filename, LEARNING_FILE_PATH);
	outfile = fopen (filename, "w");
	if (!outfile)
	{
		printf ("Error opening %s\n", filename);
		exit(0);
	}
	count=0;
	for(i= 0; i< number_of_input_patterns; i++)
	{
		goutput[i]=0;
		for(j=0; j< ensn; j++)
			goutput[i]+= output[j][i];
		goutput[i]/=ensn;
		if(goutput[i] >= CLASS_THRESHOLD)
			pred[i] = 1.0;
		else 
			pred[i] = 0.0;
		//pred[i] = (int)(goutput[i]*(maxout-minout)+0.5); 
		obs[i] =  target[i];  //(int)((target[i]-outoffset)/outscale); 
	    if(pred[i] == obs[i])
		{
			 temp = 1;
			 count++;
		}
		else
			temp = 0;
		fprintf(outfile,"%f   %f   %f    %f\n",input[i][0], obs[i], pred[i], temp);
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("training: accuracy of classification is %f  %% \n", accuracy*100.0);
	printf("exit at iteration = %d\n",iter);
	
	fclose(outfile);
	
	strcpy(filename, WEIGHT_I_H_FILE);
	outfile = fopen (filename, "w");
	if (!outfile)
	{
		printf ("Error opening %s\n", filename);
		exit(0);
	}
	for(x=0; x< ensn; x++)
	{
		for(i=0; i< input_array_size; i++)
		{
			for(j=0; j< hidden_array_size; j++)
				fprintf(outfile,"%f    ", weight_i_h[x][i][j]);
			fprintf(outfile,"\n");
		}
	}
	fclose(outfile);
	
	strcpy(filename, WEIGHT_H_O_FILE);
	outfile = fopen (filename, "w");
	if (!outfile)
	{
		printf ("Error opening %s\n", filename);
		exit(0);
	}
	for(x=0; x< ensn; x++)
	{
	   for(i=0; i< hidden_array_size; i++)
	{
		fprintf(outfile,"%f  ", weight_h_o[x][i]);
	}
	   fprintf(outfile,"\n");
	}
	fclose(outfile);
	cout << endl << "training finished" << endl;
	return;
}

void load_data(char *arg) {
	int i,x, y;
	ifstream in(arg);
	if(!in) { cout << endl << "failed to load data file" << endl; file_loaded = 0; return; }
	in >> input_array_size;
	in >> hidden_array_size;
	in >> learning_rate;
	in >> number_of_input_patterns;
	bias_array_size = hidden_array_size + 1;
	basic_memory_assign();
	initialize_net();
    for(i=0; i< ensn; i++)
	{
		for(x=0; x<bias_array_size; x++)  in >> bias[i][x];
		for(x=0; x<input_array_size; x++) { 
			for(y=0; y< hidden_array_size; y++)
			      in >> weight_i_h[i][x][y];
		}
		for(x=0; x<hidden_array_size; x++) { 
			in >> weight_h_o[i][x];
		}
		for(x=0; x<number_of_input_patterns; x++) {
			for(y=0; y< IN_ARRAY_SIZE; y++)
			in >> input[x][y];
		}
		for(x=0; x<number_of_input_patterns; x++) {
			in >> target[x];
		}
	}
	in.close();
	cout << endl << "data loaded" << endl;
	return;
}

void Initialization(int pattern_count)
{
	int i,j,x;
	input_array_size = IN_ARRAY_SIZE;  //7;    //4    //8;
	hidden_array_size = HD_ARRAY_SIZE;   //4;   //5;
	learning_rate = LEARN_RATE;  //original 0.5
	number_of_input_patterns = pattern_count;
	bias_array_size = hidden_array_size + 1;
	basic_memory_assign();
	initialize_net();
	
    for(x=0; x< NUM_ENS+3; x++)
	{
		//added by Apr 26 to test the sensitivity of initial weights
		/*if(x < NUM_ENS)
			srand(x);
		else
			srand(x-NUM_ENS);*/

		for(i=0; i<bias_array_size; i++)
			bias[x][i] = 0.00;
		for(i=0; i<input_array_size; i++) { 
			for(j=0; j<hidden_array_size; j++) 
				weight_i_h[x][i][j] = (double)rand()/2000000.0;     ///20000.0;
		}
		for(i=0; i<hidden_array_size; i++) { 
				weight_h_o[x][i] = (double)rand()/2000000.0;    ///20000.0;
		}
	}

    if(choice_lmode == '1')
	{
		for(x=0; x< NUM_ENS+3; x++)
		{
			for(i=0; i<bias_array_size; i++)
				delta_bias[x][i] = 0.00;
			for(i=0; i<input_array_size; i++) { 
				for(j=0; j<hidden_array_size; j++) 
					delta_weight_i_h[x][i][j] = 0.0;
			}
			for(i=0; i<hidden_array_size; i++) { 
					delta_weight_h_o[x][i] = 0.0;
			}
		}
	}

  	init_input_output();
}

void ens_forward_pass(int i,int pattern)
{
	register double temp=0;
	register int x,y;


	// INPUT -> HIDDEN
	for(y=0; y<hidden_array_size; y++) {
		for(x=0; x<input_array_size; x++) {
			temp += (input[pattern][x] * weight_i_h[i][x][y]);
		}
		hidden[i][y] = sigmoid(temp + bias[i][y]);
		if(hidden[i][y] < hidden_min)
			hidden_min = hidden[i][y];
		if(hidden[i][y] > hidden_max)
			hidden_max = hidden[i][y];
		//hidden[y] = temp + bias[y];
		temp = 0;
	}

	// HIDDEN -> OUTPUT
	for(x=0; x<hidden_array_size; x++) {
		temp += (hidden[i][x] * weight_h_o[i][x]);
	}
	output[i][pattern] = sigmoid(temp + bias[i][hidden_array_size]);
	if(output[i][pattern] < output_min)
		output_min = output[i][pattern];
	if(output[i][pattern] > output_max)
		output_max = output[i][pattern];

	//output[pattern][y] = temp + bias[y + hidden_array_size];
	temp = 0;

	return;
}



void ens_backward_pass(int i,int pattern, double lambda)
{
	register int x, y;
	register double temp = 0;

	// COMPUTE ERRORSIGNAL FOR OUTPUT UNITS
	//errorsignal_output[x] = (target[pattern][x] - output[pattern][x]);  //original
	errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * ((1-lambda)*(target[pattern]-output[i][pattern])+lambda*(target[pattern]-goutput[pattern])); //NCL version
	//errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (target[pattern] - output[i][pattern]);   //normal version

	// COMPUTE ERRORSIGNAL FOR HIDDEN UNITS
	for(x=0; x<hidden_array_size; x++) {
		temp += (errorsignal_output[i] * weight_h_o[i][x]);
	errorsignal_hidden[i][x] = hidden[i][x] * (1-hidden[i][x]) * temp;
	temp = 0.0;
	}

	// ADJUST WEIGHTS OF CONNECTIONS FROM HIDDEN TO OUTPUT UNITS
	double length = 0.0;
	for (x=0; x<hidden_array_size; x++) {
		length += hidden[i][x]*hidden[i][x];
	}
	if (length<=0.1) length = 0.1;
	for(x=0; x<hidden_array_size; x++) {
		//weight_h_o[x][y] += (learning_rate * errorsignal_output[y] *hidden[x]/length);  //original
		weight_h_o[i][x]+=learning_rate * errorsignal_output[i]*hidden[i][x];
	}

	// ADJUST BIASES OF HIDDEN UNITS
	for(x=hidden_array_size; x<bias_array_size; x++) {
		//bias[x] += (learning_rate * errorsignal_output[x] / length);
		bias[i][x] += 0.0; // learning_rate * errorsignal_output[i]/length; //updated Jan 24 --- for classification problems, no bias needed in h-o layer
	}

	// ADJUST WEIGHTS OF CONNECTIONS FROM INPUT TO HIDDEN UNITS
	length = 0.0;
	for (x=0; x<input_array_size; x++) {
		length += input[pattern][x]*input[pattern][x];
	}
	if (length<=0.1) length = 0.1;
	for(x=0; x<input_array_size; x++) {
		for(y=0; y<hidden_array_size; y++) {
			//weight_i_h[x][y] += (learning_rate * errorsignal_hidden[y]*input[pattern][x]/length); //original 
			weight_i_h[i][x][y]+=learning_rate*errorsignal_hidden[i][y] * 
				input[pattern][x];
		}
	}

	// ADJUST BIASES FOR OUTPUT UNITS
	for(x=0; x<hidden_array_size; x++) {
		//bias[x] += (learning_rate * errorsignal_hidden[x] / length); //original
		bias[i][x] += learning_rate * errorsignal_hidden[i][x]; 
	}

	return;
}

//batch mode learning
void ens_backward_pass_batch(int i,int pattern)
{
	register int x, y;
	register double temp = 0;

	// COMPUTE ERRORSIGNAL FOR OUTPUT UNITS
	//errorsignal_output[x] = (target[pattern][x] - output[pattern][x]);  //original
	errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (target[pattern] - output[i][pattern]);

	// COMPUTE ERRORSIGNAL FOR HIDDEN UNITS
	for(x=0; x<hidden_array_size; x++) {
		temp += (errorsignal_output[i] * weight_h_o[i][x]);
	errorsignal_hidden[i][x] = hidden[i][x] * (1-hidden[i][x]) * temp;
	temp = 0.0;
	}

	// ADJUST WEIGHTS OF CONNECTIONS FROM HIDDEN TO OUTPUT UNITS
	double length = 0.0;
	for (x=0; x<hidden_array_size; x++) {
		length += hidden[i][x]*hidden[i][x];
	}
	if (length<=0.1) length = 0.1;
	for(x=0; x<hidden_array_size; x++) {
		//weight_h_o[x][y] += (learning_rate * errorsignal_output[y] *hidden[x]/length);  //original
		delta_weight_h_o[i][x]+=learning_rate * errorsignal_output[i]*hidden[i][x];
	}

	// ADJUST BIASES OF HIDDEN UNITS
	for(x=hidden_array_size; x<bias_array_size; x++) {
		//bias[x] += (learning_rate * errorsignal_output[x] / length);
		delta_bias[i][x] += 0.0;  //learning_rate * errorsignal_output[i];   //updated Jan 24 --- for classification problems, no bias needed in h-o layer
	}

	// ADJUST WEIGHTS OF CONNECTIONS FROM INPUT TO HIDDEN UNITS
	length = 0.0;
	for (x=0; x<input_array_size; x++) {
		length += input[pattern][x]*input[pattern][x];
	}
	if (length<=0.1) length = 0.1;
	for(x=0; x<input_array_size; x++) {
		for(y=0; y<hidden_array_size; y++) {
			//weight_i_h[x][y] += (learning_rate * errorsignal_hidden[y]*input[pattern][x]/length); //original 
			delta_weight_i_h[i][x][y]+=learning_rate*errorsignal_hidden[i][y] * 
				input[pattern][x];
		}
	}

	// ADJUST BIASES FOR OUTPUT UNITS
	for(x=0; x<hidden_array_size; x++) {
		//bias[x] += (learning_rate * errorsignal_hidden[x] / length); //original
		delta_bias[i][x] += learning_rate * errorsignal_hidden[i][x];
	}

	return;
}

//change this function in order to make the criteria of learning not so strict
int compare_output_to_target()
{
	register int i,y,ytemp=0,count=0;
	register double temp;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	double pred[TRAINING_DATA_SIZE], obs[TRAINING_DATA_SIZE];
	
	for(y= 0 ; y < number_of_input_patterns; y++) {
		//pred[y] = rev_sigmoid(1.25*(output[y][0]-0.1)); 
		//obs[y] = rev_sigmoid(1.25*(target[y][0]-0.1)); 
		//pred[y] = (int)((output[y][0]-outoffset)/outscale+0.5); 
		goutput[y]=0;
		for(i=0; i< ensn; i++)
			goutput[y]+= output[i][y];
		goutput[y]/=ensn;
		if(goutput[y] >= CLASS_THRESHOLD)
			pred[y] = 1.0;
		else
			pred[y] = 0.0;
		obs[y] = target[y];     //(int)((target[y]-outoffset)/outscale);  //target[y];
		if(obs[y] == pred[y])
		{
			temp = 1;count++;
		}
		else temp=0;
	}
	printf("output: min  %f  max  %f range  %f \n", output_min, output_max, output_range);
	printf("hidden: min  %f  max  %f range  %f \n", hidden_min, hidden_max, hidden_range);
	printf("training: error rate is %f %% \n", 100*(1-(double)count/(double)number_of_input_patterns));
	printf("iter: %d \n", iter);
	printf("_____________________________________\n");
	
	if(iter >= MAX_ITER)
		return 1;
	else
	   return 0;
}

void save_data(char *argres) {
	int x, y, z;
	ofstream out;    
	out.open(argres);
	if(!out) { cout << endl << "failed to save file" << endl; return; }
	out << input_array_size << endl;
	out << hidden_array_size << endl;
	out << learning_rate << endl;
	out << number_of_input_patterns << endl << endl;
	for(z=0; z< NUM_ENS; z++)
	{
		for(x=0; x<bias_array_size; x++) out << bias[z][x] << ' ';
		out << endl << endl;
		for(x=0; x<input_array_size; x++) {
			for(y=0; y<hidden_array_size; y++) out << weight_i_h[z][x][y] << ' ';
		}
		out << endl << endl;
		for(x=0; x<hidden_array_size; x++) {
			out << weight_h_o[z][x] << ' ';
		}
	}
		out << endl << endl;
		for(x=0; x<number_of_input_patterns; x++) {
			for(y=0; y<input_array_size; y++) out << input[x][y] << ' ';
			out << endl;
		}
		out << endl;
		for(x=0; x<number_of_input_patterns; x++) {
			out << target[x] << ' ';
			out << endl;
		}
	out.close();
	cout << endl << "data saved" << endl;
	return;
}    

void change_learning_rate()
{
	if (file_loaded == 0)
	{
		cout << endl
			 << "there is no data loaded into memory"
			 << endl;
		return;
	}
	cout << endl << "actual learning rate: " << learning_rate << " new value: ";
	cin >> learning_rate;
	return;
}

void clear_memory()
{
	int x,y;
	for(x=0; x<number_of_input_patterns; x++)
	{
		delete [] input[x];
	}
	delete [] input;
    for(x=0; x< hidden_array_size; x++)
	{
		delete [] hidden[x];
	}
	delete [] hidden;
	for(x=0; x< number_of_input_patterns; x++)
	{
		delete [] output[x];
	}
	delete [] output;
	delete [] target;
	for(x=0; x< bias_array_size; x++)
	{
		delete [] bias[x];
	}
	delete [] bias;
	for(x=0; x< input_array_size; x++)
		for(y=0; y< hidden_array_size; y++)
			delete [] weight_i_h[x][y];
	for(x=0; x< input_array_size; x++)
	{
		delete [] weight_i_h[x];
	}
	delete [] weight_i_h;
	
	for(x=0; x< hidden_array_size; x++)
	{
		delete [] weight_h_o[x];
	}
	delete [] weight_h_o;
	for(x=0; x< hidden_array_size; x++)
	{
		delete [] errorsignal_hidden[x];
	}
	delete [] errorsignal_hidden;
	delete [] errorsignal_output;
	file_loaded = 0;
	return;
}

void validate()
{
	register int x,i;
	register double temp, error = 0.0, accuracy;
	int count = 0;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	CE = 0.0;
	double pred[VALIDATION_DATA_SIZE], obs[VALIDATION_DATA_SIZE];
	
   	number_of_input_patterns = VALIDATION_DATA_SIZE;

	delete []hidden;
	hidden_array_size = HD_ARRAY_SIZE;

	hidden = new double *[ensn];
	for(x=0; x< ensn; x++)
	{
		hidden[x] = new double [hidden_array_size];
		if(!hidden[x]) { cout << endl << "memory problem!"; exit(1); }
	}

	for(pattern = TRAINING_DATA_SIZE; pattern< TRAINING_DATA_SIZE+number_of_input_patterns; pattern++)
	{
        for(x=0; x< ensn; x++)
		   ens_forward_pass(x,pattern);
	}

	FILE *outfile1;
	strcpy(filename, VALIDATION_OUT_FILE_PATH1);
	outfile1 = fopen (filename, "w");
	if (!outfile1)
	{
		printf ("Error opening %s\n", filename);
		exit(0);
	}

	for(x= 0; x< number_of_input_patterns; x++)
	{
		goutput[x+TRAINING_DATA_SIZE]=0;
		for(i=0; i< ensn; i++)
			goutput[x+TRAINING_DATA_SIZE]+=output[i][x+TRAINING_DATA_SIZE];
		goutput[x+TRAINING_DATA_SIZE]/=ensn;
		if(goutput[x+TRAINING_DATA_SIZE] >= CLASS_THRESHOLD)
			pred[x] = 1.0;
		else
			pred[x] = 0.0;
		//pred[x] = (int)((goutput[x+TRAINING_DATA_SIZE]-outoffset)/outscale+0.5); 
		obs[x] = target[x+TRAINING_DATA_SIZE];   //(int)((target[x+TRAINING_DATA_SIZE]-outoffset)/outscale);
	    if(pred[x] == obs[x])
		{
			 temp = 1;
			 count++;
		}
		else
			temp = 0;
		fprintf(outfile1,"%f   %f   %f    %f\n",input[x+TRAINING_DATA_SIZE][0], obs[x], pred[x], temp);
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("validation: accuracy of classification is %f  %% \n", accuracy*100.0);
	fclose(outfile1);
	return;
}

void init_cweights() //as informative priors
{
	center_weight[0] = 1.0; center_weight[1] =1.0; center_weight[2]= 1.0;
	//center_weight[0] = 1.0; center_weight[1] =1.0; center_weight[2]= 1.0;
}

void init_center_RBF()
{
	for(int i= 0; i< CENTER_SIZE; i++) {
	    for(int j=0; j< IN_ARRAY_SIZE; j++)
		   center_RBF[i][j] = -1.0+ 2.0/(double)CENTER_SIZE*(double)i;
	}
	//scale_center_RBF();
}

void init_input_output()
{	
   for(int i=0; i< TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE; i++)
   {
     
	   for(int j=0; j< IN_ARRAY_SIZE; j++)
	   {
	      input[i][j] =  input_mushroom[i][j];  //input_aus[i][j];
		  //printf("%f   ", input[i][j]);
	   }
	 
	  target[i] =     mushroom_class[i];  //(double)aus_class[i];
	  //printf("%f \n", target[i][0]);
   }
   scale_input_output();     //for aus data, comment this line
}

void scale_input_output()
{
   int i,j,k,inmark, tempminorder, temp, tempmin;
    minout= 9999; maxout= -9999;
   for(j=0; j< IN_ARRAY_SIZE; j++)
   {
	     minin[j]= 9999;
         maxin[j]=-9999;
		 inlist[j][0] =0;
   }
   for(i=0; i< TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE; i++)
   {
	   for(j=0; j< IN_ARRAY_SIZE; j++)
	   {
	     inmark = 0;
         if(minin[j] > input[i][j]) 
			 minin[j] = input[i][j];
          for(k=1; k<= inlist[j][0]; k++)
		  {
			  if(input[i][j]== inlist[j][k])
			  {
				  inmark=1;
				  break;
			  }
		  }
		  if(inmark == 0)
		  {
			  inlist[j][0]++;
			  inlist[j][inlist[j][0]] = (int)input[i][j];		  
		  }
	   }
		 if(minout > target[i])
			 minout = target[i];
		for(j=0; j< IN_ARRAY_SIZE; j++)
		  if(maxin[j] < input[i][j]) 
			 maxin[j] = input[i][j];
		 if(maxout < target[i])
			 maxout = target[i];
   }

   //sort inlist
   for(j=0; j< IN_ARRAY_SIZE; j++)
   {
        for(i=1; i<= inlist[j][0]; i++)
		{
			tempmin = 9999;
			tempminorder = 9999;
			for(k=i; k<= inlist[j][0]; k++)
			{
                 if(inlist[j][k] < tempmin)
				 {
					 tempmin=inlist[j][k];
				     tempminorder = k;
				 }
			}
			temp = inlist[j][tempminorder];
			inlist[j][tempminorder] = inlist[j][i];
			inlist[j][i] = temp;
		}
   }
   for(j=0; j< IN_ARRAY_SIZE; j++)
   {
      inscale[j] = 2.0/(double)(inlist[j][0]-1); 
      inoffset[j] = -2.0*minin[j]/(double)(inlist[j][0]-1)-1.0;  
   }
    outscale = 1.0/(maxout-minout);
   outoffset = -1.0*minout/(maxout-minout);

   for(int i=0; i< TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE; i++)
   {
	   
	   for(j=0; j< IN_ARRAY_SIZE; j++)
	   {
	      for(k=1; k< inlist[j][0]; k++)
			  if(inlist[j][k] == (int)input[i][j])
				  break;
		  if(inlist[j][0] > 1)
	            input[i][j] = inscale[j]*(k-1)-1.0;       //inscale[j]*input[i][j]+inoffset[j];
		  else
			    input[i][j] = 0.0;
		  //printf("%f ", input[i][j]);
	}
	   //printf("\n");
	   //target[i] = outscale*target[i]+outoffset;
   }
 }

void scale_center_RBF()
{
   int i,j;
   double mincen[IN_ARRAY_SIZE], maxcen[IN_ARRAY_SIZE];
   double cen_scale[IN_ARRAY_SIZE], cen_offset[IN_ARRAY_SIZE];
   for(j=0; j< IN_ARRAY_SIZE; j++)
   {
	     mincen[j]= 9999;
         maxcen[j]=-9999;
   }
   for(i=0; i< number_of_input_patterns; i++)
   {
	   for(j=0; j< IN_ARRAY_SIZE; j++)
         if(mincen[j] > center_RBF[i][j]) 
			 mincen[j] = center_RBF[i][j];
		for(j=0; j< IN_ARRAY_SIZE; j++)
		  if(maxcen[j] < center_RBF[i][j]) 
			 maxcen[j] = center_RBF[i][j];
	 }

   for(j=0; j< IN_ARRAY_SIZE; j++)
   {
      cen_scale[j] = 2.0/(maxcen[j]-mincen[j]); 
      cen_offset[j] = -2.0*mincen[j]/(maxcen[j]-mincen[j])-1.0;  
   }
  
   for(int i=0; i< number_of_input_patterns; i++)
   {
	   for(j=0; j< IN_ARRAY_SIZE; j++)
	   {
	      center_RBF[i][j] = cen_scale[j]*center_RBF[i][j]+cen_offset[j];
		  //if(j == IN_ARRAY_SIZE-1)
			  //input[i][j]*= SCALE_COFF;
	   }
	   //target[i][0] = outscale*target[i][0]+outoffset;
   }
   
}



void perturb_training_data()
{
	int i,j,parent1,parent2;
	double temp;
    for(i =0; i< number_of_input_patterns/2; i++)
	{
		parent1 = rand()%number_of_input_patterns-1;
		parent2 = rand()%number_of_input_patterns-1;

		if(parent1<0)
			parent1 =0;
		if(parent2<0)
			parent2 =0;

		//swap parent 1 and parent 2 in input
		for(j=0; j< IN_ARRAY_SIZE; j++)
		{
            temp = input[parent1][j];
			input[parent1][j] = input[parent2][j];
			input[parent2][j] = temp;
		}

	    temp = target[parent1];
		target[parent1] = target[parent2];
		target[parent2] = temp;
	}
}

void read_and_go(char* infile1, char* infile2)
{
	int i=0, j=0, k=0, x=0; 
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
	
	x=0;
	while(fgets(tempstr,MAX_LINE_LENGTH,file))
	{
		i = 0;
		firstmark = 0;
		
		for(j=0; j< LINE_THRESHOLD; j++)
			tempchar[j] = '\0';
		
		if(tempstr[0] == '"' || tempstr[0] == '#')
			continue;
		
		for(k=0; k< hidden_array_size; k++)
		{
			while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
			{
				tempchar[firstmark] = tempstr[i];
				i++;
				firstmark++;
			}
			
			weight_i_h[x][count][k] = atof(tempchar);
			
			for(j=0; j< firstmark; j++)
				tempchar[j] = '\0';
			firstmark = 0;

			while(tempstr[i] == ' ' || tempstr[i] == 9)
			{
				i++;
			}
			
		}
		count++;

		if(count >= input_array_size)
		{
            x++;
			count-=input_array_size;
		}
		
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
		
		for(k=0; k< hidden_array_size; k++)
		{
			while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
			{
				tempchar[firstmark] = tempstr[i];
				i++;
				firstmark++;
			}
			
			weight_h_o[count][k] = atof(tempchar);
			
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
	
	if(count != ensn)
		printf("weight_h_o read error! \n");
    fclose(file);
}


