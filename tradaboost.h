
#define MAX_TAB_ITER 650  //150

double p_tab[TRAINING_DATA_SIZE], w_tab[TRAINING_DATA_SIZE];
double error_tab, beta_tab;
double b_tab, sumw_tab;

void init_weight_tab();
void tradaboost();
void tab_forward_pass(int i,int pattern);
void tab_backward_pass(int i,int pattern);
int compare_output_to_target_single(int i);

void init_weight_tab()
{
	sumw_tab =0;
    for(int i=0; i< TRAINING_DATA_SIZE; i++)
	{
		sumw_tab+=w_tab[i];
	}

	for(int i=0; i< TRAINING_DATA_SIZE; i++)
	{
        p_tab[i] = w_tab[i]/sumw_tab;
	}
}

void tradaboost()
{
	int tabi,i,j,count;
	double pred[TRAINING_DATA_SIZE], obs[TRAINING_DATA_SIZE],accuracy;
	register int y, temp,x, ytemp, subsum;
    for(i=0; i< TRAINING_DATA_SIZE; i++)
	   w_tab[i] = 1.0;
	number_of_input_patterns = TRAINING_DATA_SIZE;
	b_tab = 1.0/(1.0+sqrt(2.0*log((double)DIFF_DATA_SIZE)/NUM_ENS));

	for(tabi=0; tabi< NUM_ENS; tabi++)
	{
		 iter = 0;
		 //weight initialization
         init_weight_tab();

		 //call learner
		 while(!kbhit())
		 {
			 if(choice_lmode == '2')
			 {
				 hidden_max= output_max = 0;
				 hidden_min = output_min = 9999;
				 for(y=0; y<number_of_input_patterns; y++) {
					  tab_forward_pass(tabi,y);
					  tab_backward_pass(tabi,y);
				 }
				 hidden_range = hidden_max - hidden_min;
				 output_range = output_max - output_min;
			 }
		
			 if(compare_output_to_target_single(tabi)) {
				 cout << endl << "training " << tabi << "th NN finished" << endl;
				 cout << endl << "just press a key to continue to write learning error into file" << endl;
				 char temp_char = getch();
				 break;
			 }
			 iter++;
		 }

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
			 if(output[tabi][i] >= CLASS_THRESHOLD)
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
			 fprintf(outfile,"%f   %f   %f    %f\n",input[i][0], obs[i], pred[i], temp);
		 }
		 accuracy=(double)count/(double)number_of_input_patterns;
		 printf("training: accuracy of classification is %f  %% \n", accuracy*100.0);
		 printf("TrAdaBoost iteration = %d\n",tabi+1);
		 fclose(outfile);

		 //calculate the error
		 sumw_tab = 0.0; error_tab =0.0;
		 for(j= 0; j< SAME_TRAIN_DATA_SIZE; j++)
		 {
			   if(pred[j+DIFF_DATA_SIZE] != obs[j+DIFF_DATA_SIZE])
				   error_tab+= w_tab[j+DIFF_DATA_SIZE];
               sumw_tab+= w_tab[j+DIFF_DATA_SIZE];
		 }
         error_tab/=sumw_tab;

		 //calculate beta_tab
		 beta_tab = error_tab/(1.0-error_tab);

         //update weights
		 for(j=0; j< TRAINING_DATA_SIZE; j++)
		 {
			 if((j< DIFF_DATA_SIZE) && (pred[j] != obs[j]))
			           w_tab[j]*=b_tab;
			 else if((j >= DIFF_DATA_SIZE) && (pred[j] != obs[j]))
			 {
				 if(beta_tab!=0)
				       w_tab[j]/=beta_tab;
				 else
					 w_tab[j] =0;
			 }
		 }
	}

	//validation
	double pred1[VALIDATION_DATA_SIZE], obs1[VALIDATION_DATA_SIZE];
	count =0;
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
		   tab_forward_pass(x,pattern);
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
		ytemp = 0;
		subsum = (int)(NUM_ENS/2)+1;
		for(j = NUM_ENS-1-(int)(NUM_ENS/2); j < NUM_ENS; j++)
		{
              if(output[j][x+TRAINING_DATA_SIZE] >= CLASS_THRESHOLD)
				  ytemp++;
		}
        if(ytemp >= 0.5*subsum)
		{
			goutput[x+TRAINING_DATA_SIZE] = 1.0;
			pred1[x] = 1.0;
		}
		else
		{
			goutput[x+TRAINING_DATA_SIZE] = 0.0;
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
		fprintf(outfile1,"%f   %f   %f    %d\n",input[x+TRAINING_DATA_SIZE][0], obs1[x], pred1[x], temp);
	}
	accuracy=(double)count/(double)number_of_input_patterns;
	printf("validation: accuracy of classification is %f  %% \n", accuracy*100.0);
	fclose(outfile1);

}

void tab_forward_pass(int i,int pattern)
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



void tab_backward_pass(int i,int pattern)
{
	register int x, y;
	register double temp = 0;

	// COMPUTE ERRORSIGNAL FOR OUTPUT UNITS
	//errorsignal_output[x] = (target[pattern][x] - output[pattern][x]);  //original
	errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (w_tab[pattern]*(target[pattern]-output[i][pattern])); //TrAdaBoost version
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

int compare_output_to_target_single(int i)
{
	register int y,ytemp=0,count=0;
	register double temp;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	double pred[TRAINING_DATA_SIZE], obs[TRAINING_DATA_SIZE];
	
	for(y= 0 ; y < number_of_input_patterns; y++) {
		if(output[i][y] >= CLASS_THRESHOLD)
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
	printf("training: error rate is %f %% \n", 100*(1-(double)count/(double)number_of_input_patterns));
	printf("training %d th nn using TrAdaBoost\n", i);
	printf("iter: %d \n", iter);
	printf("_____________________________________\n");
	
	if(iter >= MAX_TAB_ITER)
		return 1;
	else
	   return 0;
}
