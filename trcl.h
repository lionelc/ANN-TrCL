
#define TRCL_SAMPLE_SIZE   100   //it can be removed now, as I no longer am concerned with static TrCL
#define MAX_TRCL_ITER 150   //150
#define MAX_TRCL_BAC_ITER 150 //150
#define MAX_TRCL_AUG_ITER 150 //150

#define TRCL_BETA      -5.0  //0.75 //100000.0 //0.70   // 0.15  //0.0075    //0.15
#define REFNN_CHOICE   0   //2   //0---GenNN   1--SpecNN  2--BacNN

//optimal param: lambda = 0.51  beta=-5.0   ref=GenNN    aug_iter=150   97.47%

double base_weight_i_h[IN_ARRAY_SIZE][HD_ARRAY_SIZE], base_weight_h_o[HD_ARRAY_SIZE];
double trcl_sample_input[TRCL_SAMPLE_SIZE][IN_ARRAY_SIZE], trcl_sample_target[TRCL_SAMPLE_SIZE];
double trcl_sample_error = 0.0;
int trcl_mark = 0;

void trcl();
void bac_train(double lambda);
void inc_train(double lambda);
void aug_train(double beta, double lambda);
void gennn_train();
void specnn_train();
void comp_train(int i, double lambda);
int compare_output_to_target_bac();
int compare_output_to_target_inc();
int trcl_compare_output_to_target_bac();
int trcl_compare_output_to_target_aug();
int trcl_compare_output_to_target_specnn();
int trcl_compare_output_to_target_gennn();
void trcl_forward_pass_gen(int i,int pattern);
void trcl_backward_pass_gen(int i,int pattern, double beta, double lambda);
void trcl_forward_pass_spec(int i,int pattern);
void trcl_backward_pass_spec(int i,int pattern, double beta, double lambda);
void trcl_sampling();
void trcl_select();

//moved from corrtest.h Apr 27
void bacnn_train();
int trcl_compare_output_to_target_bacnn();
void trcl_forward_pass_bacnn(int i,int pattern);
void trcl_backward_pass_bacnn(int i,int pattern, double beta, double lambda);


//double mutual_info(double **w11, double *w12, double **w21, double *w22, int start, int end);

void trcl()
{
//	trcl_sampling();
	int k,count, trcli;
    double pred1[VALIDATION_DATA_SIZE], obs1[VALIDATION_DATA_SIZE], accuracy;
	register int temp,x;

    bac_train(NCL_LAMBDA);
	gennn_train();
	specnn_train();
	bacnn_train();
	aug_train(TRCL_BETA, NCL_LAMBDA);
    trcl_select();

	//validation
	count =0;
   	number_of_input_patterns = VALIDATION_DATA_SIZE;
  
	delete []hidden;
	hidden_array_size = HD_ARRAY_SIZE;

	hidden = new double *[ensn+3];

	for(x=0; x< ensn+3; x++)
	{
		hidden[x] = new double [hidden_array_size];
		if(!hidden[x]) { cout << endl << "memory problem!"; exit(1); }
	}

	for(pattern = TRAINING_DATA_SIZE; pattern< TRAINING_DATA_SIZE+number_of_input_patterns; pattern++)
	{
		 for(trcli=0; trcli< trcl_mark; trcli++)
           ens_forward_pass(ens_mark[trcli],pattern);
		 ens_forward_pass(ensn,pattern);
		 ens_forward_pass(ensn+1,pattern);
		 ens_forward_pass(ensn+2, pattern);
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
		for(k=0; k< trcl_mark; k++)
			goutput[x+TRAINING_DATA_SIZE]+=output[ens_mark[k]][x+TRAINING_DATA_SIZE];
		goutput[x+TRAINING_DATA_SIZE]/=trcl_mark;
		if(goutput[x+TRAINING_DATA_SIZE] >= CLASS_THRESHOLD)
		{
			pred1[x] = 1.0;
		}
		else
		{
			pred1[x] = 0.0;
		}
		//pred[x] = (int)((goutput[x+TRAINING_DATA_SIZE]-outoffset)/outscale+0.5); 
		obs1[x] = target[x+TRAINING_DATA_SIZE];   //(int)((target[x+TRAINING_DATA_SIZE]-outoffset)/outscale);
	    if(pred1[x] == obs1[x])
		{
			 temp = 1;
			 count++;
		}
		else
			temp = 0;
		//fprintf(outfile1,"%f   %f   %f    %d\n",input[x+TRAINING_DATA_SIZE][0], obs1[x], pred1[x], temp);
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("TrCL validation: accuracy of classification is %f  %% \n", accuracy*100.0);
	count=0;
	for(x= 0; x< number_of_input_patterns; x++)
	{
		if(output[0+ensn][x+TRAINING_DATA_SIZE] >= CLASS_THRESHOLD)
		{
			pred1[x] = 1.0;
		}
		else
		{
			pred1[x] = 0.0;
		}
		//pred[x] = (int)((goutput[x+TRAINING_DATA_SIZE]-outoffset)/outscale+0.5); 
	    if(pred1[x] == obs1[x])
		{
			 temp = 1;
			 count++;
		}
		else
			temp = 0;
		fprintf(outfile1,"%f   %f   %f    %d\n",input[x+TRAINING_DATA_SIZE][0], obs1[x], pred1[x], temp);
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("TrCL validation-GenNN: accuracy of classification is %f  %% \n", accuracy*100.0);
	count=0;
	for(x= 0; x< number_of_input_patterns; x++)
	{
		if(output[1+ensn][x+TRAINING_DATA_SIZE] >= CLASS_THRESHOLD)
		{
			pred1[x] = 1.0;
		}
		else
		{
			pred1[x] = 0.0;
		}
		//pred[x] = (int)((goutput[x+TRAINING_DATA_SIZE]-outoffset)/outscale+0.5); 
	    if(pred1[x] == obs1[x])
		{
			 temp = 1;
			 count++;
		}
		else
			temp = 0;
		fprintf(outfile1,"%f   %f   %f    %d\n",input[x+TRAINING_DATA_SIZE][0], obs1[x], pred1[x], temp);
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("TrCL validation-SpecNN: accuracy of classification is %f  %% \n", accuracy*100.0);
	count=0;
	for(x= 0; x< number_of_input_patterns; x++)
	{
		if(output[2+ensn][x+TRAINING_DATA_SIZE] >= CLASS_THRESHOLD)
		{
			pred1[x] = 1.0;
		}
		else
		{
			pred1[x] = 0.0;
		}
		//pred[x] = (int)((goutput[x+TRAINING_DATA_SIZE]-outoffset)/outscale+0.5); 
	    if(pred1[x] == obs1[x])
		{
			 temp = 1;
			 count++;
		}
		else
			temp = 0;
		fprintf(outfile1,"%f   %f   %f    %d\n",input[x+TRAINING_DATA_SIZE][0], obs1[x], pred1[x], temp);
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("TrCL validation-BacNN: accuracy of classification is %f  %% \n", accuracy*100.0);
	fclose(outfile1);
}

void bac_train(double lambda)
{
	int i,j,k,count;
	double pred[DIFF_DATA_SIZE], obs[DIFF_DATA_SIZE],accuracy;
	register int y, temp;
	number_of_input_patterns = DIFF_DATA_SIZE;

	//training stage 1
	iter = 0;
	for(j=0; j< MAX_TRCL_BAC_ITER; j++)
	{
		if(choice_lmode == '2')
		{
			hidden_max= output_max = 0;
			hidden_min = output_min = 9999;
			for(y=0; y<number_of_input_patterns; y++) {
				for(i=0; i< ensn; i++)   //modified Apr 27
				     ens_forward_pass(i,y);
				for(i=0; i< ensn; i++)   //modified Apr 27
				     ens_backward_pass(i,y, lambda);
			}
			hidden_range = hidden_max - hidden_min;
			output_range = output_max - output_min;
		}
		if(trcl_compare_output_to_target_bac()) {
			cout << endl << "training finished" << endl;
			cout << endl << "just press a key to continue to write learning error into file" << endl;
			char temp_char = getch();
			break;
		}
		iter++;
	}

	count=0;
	for(k= 0; k< number_of_input_patterns; k++)
	{
		goutput[k] =0;
		for(int j=0; j< ensn; j++)
			goutput[k]+=output[j][k];
		goutput[k]/= ensn;
		if(goutput[k] >= CLASS_THRESHOLD)
			pred[k] = 1.0;
		else 
			pred[k] = 0.0;
		obs[k] = target[k];  //(int)((target[i]-outoffset)/outscale);
		if(pred[k] == obs[k])
		{
			temp = 1;
			count++;
		}
		else
			temp = 0;
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("TrCL - Background training :accuracy of classification is %f  %% \n",accuracy*100.0);

}

void gennn_train()
{
	int k,count;
	double pred[TRAINING_DATA_SIZE], obs[TRAINING_DATA_SIZE],accuracy;
	register int y, temp;
	number_of_input_patterns = TRAINING_DATA_SIZE;

	//training stage 1
	iter = 0;
	while(!kbhit())
	{
		if(choice_lmode == '2')
		{
			hidden_max= output_max = 0;
			hidden_min = output_min = 9999;
			for(y=0; y<number_of_input_patterns; y++) {
				     ens_forward_pass(0+ensn,y);
				     ens_backward_pass(0+ensn,y, 0.0);
			}
			hidden_range = hidden_max - hidden_min;
			output_range = output_max - output_min;
		}
		if(trcl_compare_output_to_target_gennn()) {
			cout << endl << "training finished" << endl;
			cout << endl << "just press a key to continue to write learning error into file" << endl;
			char temp_char = getch();
			break;
		}
		iter++;
		//else
		//fprintf(learn_file,"%d    %f   %f   %f   %f\n",(int)iter, RMSE, MAE, CE, CD);
	}
	//fclose(learn_file);

	count=0;
	for(k= 0; k< number_of_input_patterns; k++)
	{
		if(output[0+ensn][k] >= CLASS_THRESHOLD)
			pred[k] = 1.0;
		else 
			pred[k] = 0.0;
		//pred[i] = (int)((goutput[i]-outoffset)/outscale+0.5); 
		obs[k] = target[k];  //(int)((target[i]-outoffset)/outscale);
		if(pred[k] == obs[k])
		{
			temp = 1;
			count++;
		}
		else
			temp = 0;
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("TrCL - GenNN :accuracy of classification is %f  %% \n",accuracy*100.0);
}

void specnn_train()
{
	int k,count;
	double pred[TRAINING_DATA_SIZE], obs[TRAINING_DATA_SIZE],accuracy;
	register int y, temp;
	number_of_input_patterns = SAME_TRAIN_DATA_SIZE;

	//training stage 1
	iter = 0;
	while(!kbhit())
	{
		if(choice_lmode == '2')
		{
			hidden_max= output_max = 0;
			hidden_min = output_min = 9999;
			for(y=DIFF_DATA_SIZE; y<number_of_input_patterns+DIFF_DATA_SIZE; y++) {
				     ens_forward_pass(1+ensn,y);
				     ens_backward_pass(1+ensn,y, 0.0);
			}
			hidden_range = hidden_max - hidden_min;
			output_range = output_max - output_min;
		}
		if(trcl_compare_output_to_target_specnn()) {
			cout << endl << "training finished" << endl;
			cout << endl << "just press a key to continue to write learning error into file" << endl;
			char temp_char = getch();
			break;
		}
		iter++;
		//else
		//fprintf(learn_file,"%d    %f   %f   %f   %f\n",(int)iter, RMSE, MAE, CE, CD);
	}
	//fclose(learn_file);

	count=0;
	number_of_input_patterns = TRAINING_DATA_SIZE;
	for(y=0; y<number_of_input_patterns; y++) {
		     ens_forward_pass(1+ensn,y);
	}
	for(k= 0; k< number_of_input_patterns; k++)
	{
		if(output[1+ensn][k] >= CLASS_THRESHOLD)
			pred[k] = 1.0;
		else 
			pred[k] = 0.0;
		//pred[i] = (int)((goutput[i]-outoffset)/outscale+0.5); 
		obs[k] = target[k];  //(int)((target[i]-outoffset)/outscale);
		if(pred[k] == obs[k])
		{
			temp = 1;
			count++;
		}
		else
			temp = 0;
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("TrCL - SpecNN :accuracy of classification is %f  %% \n",accuracy*100.0);
}


void aug_train(double beta, double lambda)
{

	int i,k,count, trcli;
	double pred[TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE], obs[TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE],accuracy;
	register int y, temp;

	//augmentation
	iter = 0;
	number_of_input_patterns = SAME_TRAIN_DATA_SIZE;
	//call learner 
	//FILE *learn_file;
	//strcpy(filename, LEARNING_CURVE_PATH);
	//learn_file = fopen (filename, "w");
	while(!kbhit())
	{
		if(choice_lmode == '2')
		{
			hidden_max= output_max = 0;
			hidden_min = output_min = 9999;
			for(y=DIFF_DATA_SIZE; y<number_of_input_patterns+DIFF_DATA_SIZE; y++) {
				for(trcli=0; trcli< ensn; trcli++)
				{
					if(REFNN_CHOICE ==1)
				      trcl_forward_pass_spec(trcli,y);
					else if(REFNN_CHOICE ==0)
				      trcl_forward_pass_gen(trcli,y);
					else if(REFNN_CHOICE ==2)
					  trcl_forward_pass_bacnn(trcli,y);
				}
				for(trcli=0; trcli< ensn; trcli++)
				{
					if(REFNN_CHOICE == 1)
				      trcl_backward_pass_spec(trcli,y, beta, lambda);
					else if(REFNN_CHOICE ==0)
					  trcl_backward_pass_gen(trcli,y, beta, lambda);
					else if(REFNN_CHOICE == 2)
					  trcl_backward_pass_bacnn(trcli,y, beta, lambda);
				}
			}
			hidden_range = hidden_max - hidden_min;
			output_range = output_max - output_min;
		}
		/*if (!learn_file)
		{
		printf ("Error opening %s\n", filename);
		exit(0);
		}*/

		if(trcl_compare_output_to_target_aug()) {
			cout << endl << "I training finished total:" << trcli<<endl;
			cout << endl << "just press a key to continue to write learning error into file" << endl;
			char temp_char = getch();
			break;
		}
		iter++;
		//else
		//fprintf(learn_file,"%d    %f   %f   %f   %f\n",(int)iter, RMSE, MAE, CE, CD);
	}
	//fclose(learn_file);

	count=0;

	for(i= DIFF_DATA_SIZE; i< number_of_input_patterns+DIFF_DATA_SIZE; i++)
	{
		goutput[i]=0;
		for(k=0; k< ensn; k++)
			goutput[i]+=output[k][i];
		goutput[i]/=ensn;
		if(goutput[i] >= CLASS_THRESHOLD)
			pred[i] = 1.0;
		else 
			pred[i] = 0.0;
		//pred[i] = (int)((goutput[i]-outoffset)/outscale+0.5); 
		obs[i] = target[i];  //(int)((target[i]-outoffset)/outscale);
		if(pred[i] == obs[i])
		{
			temp = 1;
			count++;
		}
		else
			temp = 0;
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("TrCL-Augmentation :accuracy of classification is %f  %% \n", accuracy*100.0);
}



int compare_output_to_target_bac()
{
	register int y,ytemp=0,count=0;
	register double temp;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	double pred[DIFF_DATA_SIZE], obs[DIFF_DATA_SIZE];
	
	for(y= 0 ; y < number_of_input_patterns; y++) {
		goutput[y] =0;
		for(int j=0; j< ensn; j++)
			goutput[y]+=output[j][y];
		goutput[y]/=ensn;
		if(goutput[y] >= CLASS_THRESHOLD)
			pred[y] = 1.0;
		else
			pred[y] = 0.0;
		obs[y] = target[y];  //(int)((target[y][0]-outoffset)/outscale); 
		if(obs[y] == pred[y])
		{
			temp = 1;count++;
		}
		else temp=0;
	}
	printf("output: min  %f  max  %f range  %f \n", output_min, output_max, output_range);
	printf("hidden: min  %f  max  %f range  %f \n", hidden_min, hidden_max, hidden_range);
	printf("training I: error rate is %f %% \n", 100*(1-(double)count/(double)number_of_input_patterns));
	printf("iter: %d \n", iter);
	printf("_____________________________________\n");
	
	if(iter >= MAX_TRCL_ITER)
		return 1;
	else
	   return 0;
}

void trcl_forward_pass_spec(int i,int pattern)
{
	register double temp=0;
	register int x,y;

    trcl_sample_error = 0.0;
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

	//compare to GenNN, as the (0+NUM_ENS)th NN in the ensemble
	//an alternative option is to compare to SpecNN
	for(y=0; y<hidden_array_size; y++) {
		for(x=0; x<input_array_size; x++) {
			temp += (input[pattern][x] * weight_i_h[1+ensn][x][y]);
		}
		//hidden[0+ensn][y] = sigmoid(temp + bias[0+ensn][y]);
		hidden[1+ensn][y] = sigmoid(temp + bias[1+ensn][y]);
		//hidden[y] = temp + bias[y];
		temp = 0;
	}

	// HIDDEN -> OUTPUT
	for(x=0; x<hidden_array_size; x++) {
		temp += (hidden[1+ensn][x] * weight_h_o[1+ensn][x]);
	}
	//output[0+ensn][pattern] = sigmoid(temp + bias[0][hidden_array_size]);
	output[1+ensn][pattern] = sigmoid(temp + bias[1+ensn][hidden_array_size]);

	return;
}



void trcl_backward_pass_spec(int i,int pattern, double beta, double lambda)
{
	register int x, y;
	register double temp = 0, judge =0;

	// COMPUTE ERRORSIGNAL FOR OUTPUT UNITS
	//errorsignal_output[x] = (target[pattern][x] - output[pattern][x]);  //original
	//if(output[0+ensn][pattern] >= 0.50 && target[pattern] >= 0.50)
	
	if(output[1+ensn][pattern] >= CLASS_THRESHOLD && target[pattern] >= CLASS_THRESHOLD)
		judge = 1;
	else if(output[1+ensn][pattern] < CLASS_THRESHOLD && target[pattern]< CLASS_THRESHOLD)
	//else if(output[0+ensn][pattern] <0.50 && target[pattern]<0.50)
		judge = 1;
	if(!judge)
	   errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern])+beta*(output[1+ensn][pattern]-goutput[pattern])); //editted Jan 29
	   //errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern])+beta*(output[0+ensn][pattern]-goutput[pattern])); //editted Jan 29
	   //errorsignal_output[i] = goutput[pattern] * (1-goutput[pattern]) * ((1-beta)*(lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern]))+beta*(output[0+ensn][pattern]-goutput[pattern])); //editted Jan 29
	else
       errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern])); 
	   //errorsignal_output[i] = goutput[pattern] * (1-goutput[pattern]) * (lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern])); 
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

void trcl_forward_pass_bacnn(int i,int pattern)
{
	register double temp=0;
	register int x,y;

    trcl_sample_error = 0.0;
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

	//compare to GenNN, as the (0+NUM_ENS)th NN in the ensemble
	//an alternative option is to compare to SpecNN
	for(y=0; y<hidden_array_size; y++) {
		for(x=0; x<input_array_size; x++) {
			temp += (input[pattern][x] * weight_i_h[2+ensn][x][y]);
		}
		//hidden[0+ensn][y] = sigmoid(temp + bias[0+ensn][y]);
		hidden[2+ensn][y] = sigmoid(temp + bias[2+ensn][y]);
		//hidden[y] = temp + bias[y];
		temp = 0;
	}

	// HIDDEN -> OUTPUT
	for(x=0; x<hidden_array_size; x++) {
		temp += (hidden[2+ensn][x] * weight_h_o[2+ensn][x]);
	}
	//output[0+ensn][pattern] = sigmoid(temp + bias[0][hidden_array_size]);
	output[2+ensn][pattern] = sigmoid(temp + bias[2+ensn][hidden_array_size]);

	return;
}



void trcl_backward_pass_bacnn(int i,int pattern, double beta, double lambda)
{
	register int x, y;
	register double temp = 0, judge =0;

	// COMPUTE ERRORSIGNAL FOR OUTPUT UNITS
	//errorsignal_output[x] = (target[pattern][x] - output[pattern][x]);  //original
	//if(output[0+ensn][pattern] >= 0.50 && target[pattern] >= 0.50)
	
	if(output[2+ensn][pattern] >= CLASS_THRESHOLD && target[pattern] >= CLASS_THRESHOLD)
		judge = 1;
	else if(output[2+ensn][pattern] < CLASS_THRESHOLD && target[pattern]< CLASS_THRESHOLD)
	//else if(output[0+ensn][pattern] <0.50 && target[pattern]<0.50)
		judge = 1;
	if(!judge)
	   errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern])+beta*(output[2+ensn][pattern]-goutput[pattern])); //editted Jan 29
	   //errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern])+beta*(output[0+ensn][pattern]-goutput[pattern])); //editted Jan 29
	   //errorsignal_output[i] = goutput[pattern] * (1-goutput[pattern]) * ((1-beta)*(lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern]))+beta*(output[0+ensn][pattern]-goutput[pattern])); //editted Jan 29
	else
       errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern])); 
	   //errorsignal_output[i] = goutput[pattern] * (1-goutput[pattern]) * (lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern])); 
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

int compare_output_to_target_inc()
{
	register int y,ytemp=0,count=0;
	register double temp;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	double pred[SAME_TRAIN_DATA_SIZE], obs[SAME_TRAIN_DATA_SIZE];
	
	for(y= 0 ; y < number_of_input_patterns; y++) {
		goutput[y+DIFF_DATA_SIZE] =0;
		for(int j=0; j< ensn; j++)
			goutput[y+DIFF_DATA_SIZE]+=output[j][y+DIFF_DATA_SIZE];
		goutput[y+DIFF_DATA_SIZE]/=ensn;
		if(goutput[y+DIFF_DATA_SIZE] >= CLASS_THRESHOLD)
			pred[y] = 1.0;
		else
			pred[y] = 0.0;
		obs[y] = target[y+DIFF_DATA_SIZE];  //(int)((target[y][0]-outoffset)/outscale); 
		if(obs[y] == pred[y])
		{
			temp = 1;count++;
		}
		else temp=0;
	}
	printf("output: min  %f  max  %f range  %f \n", output_min, output_max, output_range);
	printf("hidden: min  %f  max  %f range  %f \n", hidden_min, hidden_max, hidden_range);
	printf("training II-III: error rate is %f %% \n", 100*(1-(double)count/(double)number_of_input_patterns));
	printf("iter: %d \n", iter);
	printf("_____________________________________\n");
	
	if(iter >= MAX_TRCL_ITER)
		return 1;
	else
	   return 0;
}

int trcl_compare_output_to_target_aug()
{
	register int y,ytemp=0,count=0;
	register double temp;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	double pred[SAME_TRAIN_DATA_SIZE], obs[SAME_TRAIN_DATA_SIZE];
	
	for(y= 0 ; y < number_of_input_patterns; y++) {
		goutput[y+DIFF_DATA_SIZE] =0;
		for(int j=0; j< ensn; j++)
			goutput[y+DIFF_DATA_SIZE]+=output[j][y+DIFF_DATA_SIZE];
		goutput[y+DIFF_DATA_SIZE]/=ensn;
		if(goutput[y+DIFF_DATA_SIZE] >= CLASS_THRESHOLD)
			pred[y] = 1.0;
		else
			pred[y] = 0.0;
		obs[y] = target[y+DIFF_DATA_SIZE];  //(int)((target[y][0]-outoffset)/outscale); 
		if(obs[y] == pred[y])
		{
			temp = 1;count++;
		}
		else temp=0;
	}
	printf("output: min  %f  max  %f range  %f \n", output_min, output_max, output_range);
	printf("hidden: min  %f  max  %f range  %f \n", hidden_min, hidden_max, hidden_range);
	printf("TrCL-Augmentation: error rate is %f %% \n", 100*(1-(double)count/(double)number_of_input_patterns));
	printf("iter: %d \n", iter);
	printf("_____________________________________\n");
	
	if(iter >= MAX_TRCL_AUG_ITER)
		return 1;
	else
	   return 0;
}

void inc_train(double lambda)
{

	int i,j,k,count, trcli;
	double pred[TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE], obs[TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE],accuracy;
	register int y, temp,x;
	number_of_input_patterns = DIFF_DATA_SIZE;

	//training stage 1
	iter = 0;
	//call learner
	//FILE *learn_file;
	//strcpy(filename, LEARNING_CURVE_PATH);
	//learn_file = fopen (filename, "w");
	for(j=0; j< MAX_TRCL_ITER; j++)
	{
		if(choice_lmode == '2')
		{
			hidden_max= output_max = 0;
			hidden_min = output_min = 9999;
			for(y=0; y<number_of_input_patterns; y++) {
				for(trcli=0; trcli< ensn; trcli++)
				    ens_forward_pass(trcli,y);
				for(trcli=0; trcli< ensn; trcli++)
				    ens_backward_pass(trcli,y, lambda);
			}
			hidden_range = hidden_max - hidden_min;
			output_range = output_max - output_min;
		}
		if(compare_output_to_target_bac()) {
			cout << endl << "training stage I finished" << endl;
			cout << endl << "just press a key to continue to write learning error into file" << endl;
			char temp_char = getch();
			break;
		}
		iter++;
		//else
		//fprintf(learn_file,"%d    %f   %f   %f   %f\n",(int)iter, RMSE, MAE, CE, CD);
	}
	//fclose(learn_file);

	count=0;
	for(i= 0; i< number_of_input_patterns; i++)
	{   	
		goutput[i]=0;
		for(k=0; k< ensn; k++)
			goutput[i]+=output[k][i];
		goutput[i]/=ensn;
		if(goutput[i] >= CLASS_THRESHOLD)
			pred[i] = 1.0;
		else 
			pred[i] = 0.0;
		//pred[i] = (int)((goutput[i]-outoffset)/outscale+0.5); 
		obs[i] = target[i];  //(int)((target[i]-outoffset)/outscale);
		if(pred[i] == obs[i])
		{
			temp = 1;
			count++;
		}
		else
			temp = 0;
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("incremental training I:accuracy of classification is %f  %% \n", accuracy*100.0);

	//training stage 2
	iter = 0;
	number_of_input_patterns = SAME_TRAIN_DATA_SIZE;
	//call learner 
	//FILE *learn_file;
	//strcpy(filename, LEARNING_CURVE_PATH);
	//learn_file = fopen (filename, "w");
	while(!kbhit())
	{
		if(choice_lmode == '2')
		{
			hidden_max= output_max = 0;
			hidden_min = output_min = 9999;
			for(y=DIFF_DATA_SIZE; y<number_of_input_patterns+DIFF_DATA_SIZE; y++) {
			//for(y=0; y<number_of_input_patterns; y++) {
				for(trcli=0; trcli< ensn; trcli++)
				    ens_forward_pass(trcli,y);
				for(trcli=0; trcli< ensn; trcli++)
				    ens_backward_pass(trcli,y, lambda);
			}
			hidden_range = hidden_max - hidden_min;
			output_range = output_max - output_min;
		}
		if(compare_output_to_target_inc()) {
			cout << endl << "training stage II finished" <<endl;
			cout << endl << "just press a key to continue to write learning error into file" << endl;
			char temp_char = getch();
			break;
		}
		iter++;
	}

	count=0;

	for(i= DIFF_DATA_SIZE; i< number_of_input_patterns+DIFF_DATA_SIZE; i++)
	{
		goutput[i]=0;
		for(k=0; k< ensn; k++)
			goutput[i]+=output[k][i];
		goutput[i]/=ensn;
		if(goutput[i] >= CLASS_THRESHOLD)
			pred[i] = 1.0;
		else 
			pred[i] = 0.0;
		obs[i] = target[i]; 
		if(pred[i] == obs[i])
		{
			temp = 1;
			count++;
		}
		else
			temp = 0;
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("incremental training II:accuracy of classification is %f  %% \n", accuracy*100.0);

 	//validation
	double pred1[VALIDATION_DATA_SIZE], obs1[VALIDATION_DATA_SIZE];
	//double pred1[SAME_TRAIN_DATA_SIZE], obs1[SAME_TRAIN_DATA_SIZE];
	count =0;
   	number_of_input_patterns = SAME_TRAIN_DATA_SIZE;  //VALIDATION_DATA_SIZE;

	for(x=0; x< ensn; x++)
	{
		hidden[x] = new double [hidden_array_size];
		if(!hidden[x]) { cout << endl << "memory problem!"; exit(1); }
	}

	for(pattern = TRAINING_DATA_SIZE; pattern< TRAINING_DATA_SIZE+number_of_input_patterns; pattern++)
	//for(pattern = DIFF_DATA_SIZE; pattern< number_of_input_patterns+DIFF_DATA_SIZE; pattern++)
	{
		 for(trcli=0; trcli<ensn; trcli++)
           ens_forward_pass(trcli,pattern);
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
		//goutput[x]=0;
		for(k=0; k< ensn; k++)
			//goutput[x]+=output[k][x];
		    goutput[x+TRAINING_DATA_SIZE]+=output[k][x+TRAINING_DATA_SIZE];
		//goutput[x]/=ensn;
		goutput[x+TRAINING_DATA_SIZE]/=ensn;
		if(goutput[x+TRAINING_DATA_SIZE] >= CLASS_THRESHOLD)
		{
			pred1[x] = 1.0;
		}
		else
		{
			pred1[x] = 0.0;
		}
		//pred[x] = (int)((goutput[x+TRAINING_DATA_SIZE]-outoffset)/outscale+0.5); 
		//obs1[x] = target[x];   //(int)((target[x+TRAINING_DATA_SIZE]-outoffset)/outscale);
		obs1[x] = target[x+TRAINING_DATA_SIZE];
	    if(pred1[x] == obs1[x])
		{
			 temp = 1;
			 count++;
		}
		else
			temp = 0;
		fprintf(outfile1,"%f   %f   %f    %d\n",input[x+TRAINING_DATA_SIZE][0], obs1[x], pred1[x], temp);
		//fprintf(outfile1,"%f   %f   %f    %d\n",input[x][0], obs1[x], pred1[x], temp);
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("Incremental learning validation: accuracy of classification is %f  %% \n", accuracy*100.0);
	fclose(outfile1);
}

void trcl_sampling()
{
	int j, k,pos;
	double dpos;
    int posmark[DIFF_DATA_SIZE];

	for(j=0; j< TRCL_SAMPLE_SIZE; j++)
	   posmark[j] =0;

	for(j=0; j< TRCL_SAMPLE_SIZE; j++)
	{
		dpos = (double)rand()/(double)RAND_MAX;
		pos = (int)(dpos*DIFF_DATA_SIZE);

		if(posmark[pos] == 0)
			posmark[pos] =1;
		else
		{
			while(posmark[pos] == 0)
				pos++;
		}

		for(k=0;k< IN_ARRAY_SIZE; k++)
			trcl_sample_input[j][k] = input[pos][k];
		trcl_sample_target[j] = target[pos];
	}

}

int trcl_compare_output_to_target_bac()
{
	register int y,ytemp=0,count=0;
	register double temp;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	double pred[DIFF_DATA_SIZE], obs[DIFF_DATA_SIZE];
	
	for(y= 0 ; y < number_of_input_patterns; y++) {
		goutput[y] =0;
		for(int j=0; j< ensn; j++)
			goutput[y]+=output[j][y];
		goutput[y]/=ensn-1;
		if(goutput[y] >= CLASS_THRESHOLD)
			pred[y] = 1.0;
		else
			pred[y] = 0.0;
		obs[y] = target[y];  //(int)((target[y][0]-outoffset)/outscale); 
		if(obs[y] == pred[y])
		{
			temp = 1;count++;
		}
		else temp=0;
	}
	printf("output: min  %f  max  %f range  %f \n", output_min, output_max, output_range);
	printf("hidden: min  %f  max  %f range  %f \n", hidden_min, hidden_max, hidden_range);
	printf("TrCL-Backgrond Learning: error rate is %f %% \n", 100*(1-(double)count/(double)number_of_input_patterns));
	printf("iter: %d \n", iter);
	printf("_____________________________________\n");
	
	if(iter >= MAX_TRCL_BAC_ITER)
		return 1;
	else
	   return 0;
}

int trcl_compare_output_to_target_gennn()
{
	register int y,ytemp=0,count=0;
	register double temp;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	double pred[TRAINING_DATA_SIZE], obs[TRAINING_DATA_SIZE];
	
	//for(y= 0 ; y < number_of_input_patterns; y++) {
	for(y= 0; y < number_of_input_patterns; y++) {
		if(output[0+ensn][y] >= CLASS_THRESHOLD)
			pred[y] = 1.0;
		else
			pred[y] = 0.0;
		obs[y] = target[y];  //(int)((target[y][0]-outoffset)/outscale); 
		if(obs[y] == pred[y])
		{
			temp = 1;count++;
		}
		else temp=0;
	}
	printf("output: min  %f  max  %f range  %f \n", output_min, output_max, output_range);
	printf("hidden: min  %f  max  %f range  %f \n", hidden_min, hidden_max, hidden_range);
	printf("TrCL-GenNN training: error rate is %f %% \n", 100*(1-(double)count/(double)number_of_input_patterns));
	printf("iter: %d \n", iter);
	printf("_____________________________________\n");
	
	if(iter >= MAX_TRCL_ITER)
		return 1;
	else
	   return 0;
}

int trcl_compare_output_to_target_specnn()
{
	register int y,ytemp=0,count=0;
	register double temp;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	double pred[SAME_TRAIN_DATA_SIZE], obs[SAME_TRAIN_DATA_SIZE];
	
	for(y= 0; y < number_of_input_patterns; y++) {
		if(output[1+ensn][y+DIFF_DATA_SIZE] >= CLASS_THRESHOLD)
			pred[y] = 1.0;
		else
			pred[y] = 0.0;
		obs[y] = target[y+DIFF_DATA_SIZE];  //(int)((target[y][0]-outoffset)/outscale); 
		if(obs[y] == pred[y])
		{
			temp = 1;count++;
		}
		else temp=0;
	}
	printf("output: min  %f  max  %f range  %f \n", output_min, output_max, output_range);
	printf("hidden: min  %f  max  %f range  %f \n", hidden_min, hidden_max, hidden_range);
	printf("TrCL-SpecNN training: error rate is %f %% \n", 100*(1-(double)count/(double)number_of_input_patterns));
	printf("iter: %d \n", iter);
	printf("_____________________________________\n");
	
	if(iter >= MAX_TRCL_ITER)
		return 1;
	else
	   return 0;
}

void trcl_select()
{

	int i,j,count, tcli;
	double pred1[SAME_TRAIN_DATA_SIZE], obs1[SAME_TRAIN_DATA_SIZE],accuracy[NUM_ENS+3];
	register int temp,x;
	number_of_input_patterns = SAME_TRAIN_DATA_SIZE;
    iter = 0;

	for(i=0; i< ensn+3; i++)
		ens_mark[i] = i;
  
	//validate other NNs
	count =0;
   	number_of_input_patterns = SAME_TRAIN_DATA_SIZE;

	for(x=0; x< ensn; x++)
	{
		hidden[x] = new double [hidden_array_size];
		if(!hidden[x]) { cout << endl << "memory problem!"; exit(1); }
	}

	for(pattern = DIFF_DATA_SIZE; pattern< DIFF_DATA_SIZE+number_of_input_patterns; pattern++)
	{
		 for(tcli=0; tcli<ensn+3; tcli++)
           ens_forward_pass(tcli,pattern);
	}

	for(tcli=0; tcli<ensn+3; tcli++)
	{
		FILE *outfile1;
		count =0;
		strcpy(filename, VALIDATION_OUT_FILE_PATH1);
		outfile1 = fopen (filename, "w");
		if (!outfile1)
		{
			printf ("Error opening %s\n", filename);
			exit(0);
		}

		for(x= 0; x< number_of_input_patterns; x++)
		{
			if(output[tcli][x+DIFF_DATA_SIZE] >= CLASS_THRESHOLD)
			{
				pred1[x] = 1.0;
			}
			else
			{
				pred1[x] = 0.0;
			}
			//pred[x] = (int)((goutput[x+TRAINING_DATA_SIZE]-outoffset)/outscale+0.5); 
			obs1[x] = target[x+DIFF_DATA_SIZE];   //(int)((target[x+TRAINING_DATA_SIZE]-outoffset)/outscale);
			if(pred1[x] == obs1[x])
			{
				temp = 1;
				count++;
			}
			else
				temp = 0;
			fprintf(outfile1,"%f   %f   %f    %d\n",input[x+DIFF_DATA_SIZE][0], obs1[x], pred1[x], temp);
		}
		accuracy[tcli]=(double)count/(double)number_of_input_patterns;
		printf("TrCL-selection: %d th NN - accuracy of classification is %f  %% \n", tcli, accuracy[tcli]*100.0);
		fclose(outfile1);
	}

	double tempmax; int temppos, temp1;

	//rank all the learners
	for(i =0; i< ensn; i++)
	{
		tempmax = accuracy[i];
		temppos = i;
		for(j=i+1; j< ensn; j++)
		{
            if(accuracy[j]> tempmax)
			{
				tempmax = accuracy[j];
				temppos = j;
			}
		}
		accuracy[temppos] = accuracy[i];
		temp1 = ens_mark[temppos];
        ens_mark[temppos] = ens_mark[i];
		ens_mark[i] = temp1;
		accuracy[i] = tempmax;
	}

	trcl_mark =0;

	//select all with higher accuracy than either SpecNN or both BacNN & GenNN
	int tempspec, tempgen, tempbac;
	for(i=0; i< ensn+3; i++)
	{
         if(ens_mark[i] == ensn+1)
			 tempspec = i;
		 else if(ens_mark[i] == ensn+2)
			 tempbac = i;
		 else if(ens_mark[i] == ensn+0)
			 tempgen = i;
	}
	for(i=0; i<ensn; i++)
	{
		if( (accuracy[ens_mark[i]] >= accuracy[tempspec]) && (accuracy[ens_mark[i]] >= accuracy[tempgen] && accuracy[ens_mark[i]] >= accuracy[tempbac]))
                 trcl_mark++;
	}
	if(trcl_mark > ensn)
		trcl_mark = ensn;
    
	//winner-take-all    
	//trcl_mark = 1;
	//simple average
	trcl_mark = ensn;

	if(trcl_mark == 0)
	{
		tempmax = 0.0; 
		temppos = -1;
		for(i=0; i< 3; i++)
		{
		   if(accuracy[i+ensn] > tempmax)
		   {
              tempmax = accuracy[i+ensn];
			  temppos = i;
		   }
		}
		ens_mark[0] = temppos+ensn;
		trcl_mark = 1;
	}
	printf("%d individual NNs in the final ensemble: ", trcl_mark);   //added Apr 27
	for(i=0; i< trcl_mark; i++)
	    printf("%d ", ens_mark[i]);
	printf("\n");
}

void trcl_forward_pass_gen(int i,int pattern)
{
	register double temp=0;
	register int x,y;

    trcl_sample_error = 0.0;
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

	//compare to GenNN, as the (0+NUM_ENS)th NN in the ensemble
	//an alternative option is to compare to SpecNN
	for(y=0; y<hidden_array_size; y++) {
		for(x=0; x<input_array_size; x++) {
			temp += (input[pattern][x] * weight_i_h[0+ensn][x][y]);
		}
		//hidden[0+ensn][y] = sigmoid(temp + bias[0+ensn][y]);
		hidden[0+ensn][y] = sigmoid(temp + bias[0+ensn][y]);
		//hidden[y] = temp + bias[y];
		temp = 0;
	}

	// HIDDEN -> OUTPUT
	for(x=0; x<hidden_array_size; x++) {
		temp += (hidden[0+ensn][x] * weight_h_o[0+ensn][x]);
	}
	//output[0+ensn][pattern] = sigmoid(temp + bias[0][hidden_array_size]);
	output[0+ensn][pattern] = sigmoid(temp + bias[0+ensn][hidden_array_size]);

	return;
}



void trcl_backward_pass_gen(int i,int pattern, double beta, double lambda)
{
	register int x, y;
	register double temp = 0, judge =0;

	// COMPUTE ERRORSIGNAL FOR OUTPUT UNITS
	//errorsignal_output[x] = (target[pattern][x] - output[pattern][x]);  //original
	if(output[0+ensn][pattern] >= CLASS_THRESHOLD && target[pattern] >= CLASS_THRESHOLD)	
		judge = 1;
	else if(output[0+ensn][pattern] < CLASS_THRESHOLD && target[pattern]< CLASS_THRESHOLD)
		judge = 1;
	if(!judge)  //if I change the sign of this... the result is the same as SpecNN... why?   added Apr 26
	   errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern])+beta*(output[0+ensn][pattern]-goutput[pattern])); //editted Jan 29
	   //errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern])+beta*(output[0+ensn][pattern]-goutput[pattern])); //editted Jan 29
	   //errorsignal_output[i] = goutput[pattern] * (1-goutput[pattern]) * ((1-beta)*(lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern]))+beta*(output[0+ensn][pattern]-goutput[pattern])); //editted Jan 29
	else
       errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern])); 
	   //errorsignal_output[i] = goutput[pattern] * (1-goutput[pattern]) * (lambda*(target[pattern]-goutput[pattern])+(1-lambda)*(target[pattern]-output[i][pattern])); 
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

void bacnn_train()
{
	int k,count;
	double pred[TRAINING_DATA_SIZE], obs[TRAINING_DATA_SIZE],accuracy;
	register int y, temp;
	number_of_input_patterns =DIFF_DATA_SIZE;

	//training stage 1
	iter = 0;
	while(!kbhit())
	{
		if(choice_lmode == '2')
		{
			hidden_max= output_max = 0;
			hidden_min = output_min = 9999;
			for(y=0; y<number_of_input_patterns; y++) {
				     ens_forward_pass(2+ensn,y);
				     ens_backward_pass(2+ensn,y, 0.0);
			}
			hidden_range = hidden_max - hidden_min;
			output_range = output_max - output_min;
		}
		if(trcl_compare_output_to_target_bacnn()) {
			cout << endl << "training finished" << endl;
			cout << endl << "just press a key to continue to write learning error into file" << endl;
			char temp_char = getch();
			break;
		}
		iter++;
		//else
		//fprintf(learn_file,"%d    %f   %f   %f   %f\n",(int)iter, RMSE, MAE, CE, CD);
	}
	//fclose(learn_file);

	count=0;
	number_of_input_patterns = TRAINING_DATA_SIZE;
	for(y=0; y<number_of_input_patterns; y++) {
		     ens_forward_pass(2+ensn,y);
	}
	for(k= 0; k< number_of_input_patterns; k++)
	{
		if(output[2+ensn][k] >= CLASS_THRESHOLD)
			pred[k] = 1.0;
		else 
			pred[k] = 0.0;
		//pred[i] = (int)((goutput[i]-outoffset)/outscale+0.5); 
		obs[k] = target[k];  //(int)((target[i]-outoffset)/outscale);
		if(pred[k] == obs[k])
		{
			temp = 1;
			count++;
		}
		else
			temp = 0;
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("TrCL - BacNN :accuracy of classification is %f  %% \n",accuracy*100.0);

}

int trcl_compare_output_to_target_bacnn()
{
	register int y,ytemp=0,count=0;
	register double temp;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	double pred[DIFF_DATA_SIZE], obs[DIFF_DATA_SIZE];
	
	for(y= 0; y < number_of_input_patterns; y++) {
		if(output[2+ensn][y] >= CLASS_THRESHOLD)
			pred[y] = 1.0;
		else
			pred[y] = 0.0;
		obs[y] = target[y];  //(int)((target[y][0]-outoffset)/outscale); 
		if(obs[y] == pred[y])
		{
			temp = 1;count++;
		}
		else temp=0;
	}
	printf("output: min  %f  max  %f range  %f \n", output_min, output_max, output_range);
	printf("hidden: min  %f  max  %f range  %f \n", hidden_min, hidden_max, hidden_range);
	printf("TrCL-BacNN training: error rate is %f %% \n", 100*(1-(double)count/(double)number_of_input_patterns));
	printf("iter: %d \n", iter);
	printf("_____________________________________\n");
	
	if(iter >= MAX_TRCL_ITER)
		return 1;
	else
	   return 0;
}