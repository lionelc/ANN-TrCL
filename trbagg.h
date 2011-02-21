#define MAX_TBG_ITER 150
#define BAGG_SIZE_MEM   3500  //600
#define BAGG_SIZE_VAR  1200  //200
#define BAGG_TARGET_PROB   0.2
#define BAGG_COPY_PROB     0.3

double ***bagg_input; //[NUM_ENS][BAG_SIZE][IN_ARRAY_SIZE];
double **bagg_target;  //[NUM_ENS][BAG_SIZE];
int    *bagg_size;
int    bagg_mark;

void trbagg();
void bagg_init_data_memory();
void bagg_producing();
void bagg_train();
void bagg_select();
int compare_output_to_target_tbg(int i);
void bagg_forward_pass(int i,int pattern);
void bagg_backward_pass(int i,int pattern);

void trbagg()
{
	//validation
	int j,k,count, tbgi;
	register int x, y,ytemp=0;
	double pred1[VALIDATION_DATA_SIZE], obs1[VALIDATION_DATA_SIZE],accuracy;
	count =0;
   
	hidden_array_size = HD_ARRAY_SIZE;

	bagg_producing();
	bagg_train();
	bagg_select();

	number_of_input_patterns = VALIDATION_DATA_SIZE;
	for(pattern = TRAINING_DATA_SIZE; pattern< TRAINING_DATA_SIZE+number_of_input_patterns; pattern++)
	{
		 for(tbgi=0; tbgi< bagg_mark; tbgi++)  //ensn: many better than all?
           ens_forward_pass(ens_mark[tbgi],pattern);
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
		/*int temp; ytemp=0;
		for(j = 0; j < bagg_mark; j++)
		{
              if(output[ens_mark[j]][x+TRAINING_DATA_SIZE] >= 0.50)
				  ytemp++;
		}
        if(ytemp >= (int)(bagg_mark/2+0.5))
		{
			goutput[x+TRAINING_DATA_SIZE] = 1.0;
			pred1[x] = 1.0;
		}
		else
		{
			goutput[x+TRAINING_DATA_SIZE] = 0.0;
			pred1[x] = 0.0;
		}*/
		int temp = bagg_mark;  //(int)((double)ensn*0.35);
		goutput[x+TRAINING_DATA_SIZE]=0;
		for(k=0; k< temp; k++)  
			goutput[x+TRAINING_DATA_SIZE]+=output[ens_mark[k]][x+TRAINING_DATA_SIZE];
		goutput[x+TRAINING_DATA_SIZE]/=temp;

		if(goutput[x+TRAINING_DATA_SIZE] >= CLASS_THRESHOLD)
			pred1[x] = 1.0;
		else
			pred1[x] = 0.0;
	
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
	printf("TrBagg final validation: accuracy of classification is %f  %% \n", accuracy*100.0);

    for(y=0; y< bagg_mark; y++)
	{
		count =0;
		for(x= 0; x< number_of_input_patterns; x++)
		{
			int temp; ytemp=0;
			
			if(output[ens_mark[y]][x+TRAINING_DATA_SIZE] >= CLASS_THRESHOLD) 
			{
				pred1[x] = 1.0;
			}
			else
			{
				pred1[x] = 0.0;
			}
			/*int temp = (int)((double)ensn*0.35);
			goutput[x+TRAINING_DATA_SIZE]=0;
			for(k=0; k< temp; k++)  
			goutput[x+TRAINING_DATA_SIZE]+=output[ens_mark[k]][x+TRAINING_DATA_SIZE];
			goutput[x+TRAINING_DATA_SIZE]/=temp;*/

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
		printf("TrBagg final validation-%d th NN: accuracy of classification is %f  %% \n", ens_mark[y], accuracy*100.0);
	}
	fclose(outfile1);
}

void bagg_init_data_memory()
{
	int x,y;
     bagg_input = new double **[NUM_ENS]; 
	if(!bagg_input) { cout << endl << "memory problem!"; exit(1); }
	for(x=0; x< NUM_ENS; x++)
	{
		bagg_input[x] = new double *[BAGG_SIZE_MEM];
		if(!bagg_input[x]) { cout << endl << "memory problem!"; exit(1); }
	}
    for(x=0; x< NUM_ENS;  x++)
		for(y=0; y< BAGG_SIZE_MEM; y++)
		{
           bagg_input[x][y] = new double [IN_ARRAY_SIZE];
		   if(!bagg_input[x][y]) { cout << endl << "memory problem!"; exit(1); }
		}

	bagg_target = new double *[NUM_ENS];
	if(!bagg_target) { cout << endl << "memory problem!"; exit(1); }
	for(x=0; x< NUM_ENS; x++)
	{
		bagg_target[x] = new double [BAGG_SIZE_MEM];
		if(!bagg_target[x]) { cout << endl << "memory problem!"; exit(1); }
	}

	bagg_size = new int [NUM_ENS];
}

void bagg_producing()
{
   int i,j, k,pos, posc,count, blabel, usize, clabel;
   double dpos;
   double var;
   int posmark[TRAINING_DATA_SIZE];
   bagg_init_data_memory();

   //init bag size 
   for(i=1; i< ensn; i++)
   {
       var = (double)BAGG_SIZE_VAR*(double)rand()/(double)RAND_MAX;
	   bagg_size[i] = BAGG_SIZE_MEM-(int)var;
   }

   for(i=0; i< TRAINING_DATA_SIZE; i++)
	   posmark[i] =0;
   //select and copy to each bag
   for(i=1; i< ensn; i++)
   {
       count =0;
	   usize = (int)(0.632*(double)bagg_size[i]);

	   //fill unique patterns
	   for(j=0; j< usize; j++)
	   {
		   if((double)rand()/(double)RAND_MAX <= BAGG_TARGET_PROB)
			   blabel = 1;
		   else
			   blabel = 0;

		   dpos = (double)rand()/(double)RAND_MAX;
		   if(blabel)
			   pos = DIFF_DATA_SIZE+(int)(dpos*SAME_TRAIN_DATA_SIZE);
		   else
			   pos = (int)(dpos*DIFF_DATA_SIZE);

		   if(posmark[pos] == 0)
		        posmark[pos] =1;
		   else
		   {
			   while(posmark[pos] == 0)
				   pos++;
		   }

          for(k=0;k< IN_ARRAY_SIZE; k++)
		   bagg_input[i][j][k] = input[pos][k];
		  bagg_target[i][j] = target[pos];
	   }

	   for(j=usize; j< bagg_size[i]; j++)
	   {
           if((double)rand()/(double)RAND_MAX <= BAGG_COPY_PROB)
			   clabel = 1;
		   else
			   clabel = 0;

           dpos = (double)rand()/(double)RAND_MAX;
		   if(clabel)
			   posc = DIFF_DATA_SIZE+(int)(dpos*SAME_TRAIN_DATA_SIZE);
		   else
			   posc = (int)(dpos*DIFF_DATA_SIZE);

           if(posmark[posc] == 0)
		   {
			   while(clabel==0 && posmark[posc] == 0 && posc<DIFF_DATA_SIZE)
				   posc++;

			   if(posc >= DIFF_DATA_SIZE && clabel == 0)
				   posc-=DIFF_DATA_SIZE;

                while(clabel==1 && posmark[posc] == 0 && posc<TRAINING_DATA_SIZE)
				   posc++;

                if(posc >= TRAINING_DATA_SIZE && clabel == 1)
				   posc-=SAME_TRAIN_DATA_SIZE;
		   }
		 for(k=0;k< IN_ARRAY_SIZE; k++)
		   bagg_input[i][j][k] = input[posc][k];
		  bagg_target[i][j] = target[posc];
	   }
   }
}

void bagg_train()
{
	int tbgi;
	register int y;

	//training stage 1
	for(tbgi=1; tbgi< ensn; tbgi++)
	{
		iter =0;
		number_of_input_patterns = bagg_size[tbgi];
		while(!kbhit())
		{
			if(choice_lmode == '2')
			{
				hidden_max= output_max = 0;
				hidden_min = output_min = 9999;
				for(y=0; y<number_of_input_patterns; y++) {

					bagg_forward_pass(tbgi,y);
					bagg_backward_pass(tbgi,y);
				}
				hidden_range = hidden_max - hidden_min;
				output_range = output_max - output_min;
			}
		
			if(compare_output_to_target_tbg(tbgi)) {
					cout << endl << "training finished" << endl;
					cout << endl << "just press a key to continue to write learning error into file" << endl;
					char temp_char = getch();
					break;
				}
				iter++;
			}
	}
}

void bagg_select()
{
	int i,j,count, tbgi;
	double pred[SAME_TRAIN_DATA_SIZE], obs[SAME_TRAIN_DATA_SIZE],accuracy[NUM_ENS];
	register int y, temp,x;
	number_of_input_patterns = SAME_TRAIN_DATA_SIZE;
    iter = 0;

	for(i=0; i< ensn; i++)
		ens_mark[i] = i;
   //train the nn only from same-train data (the order in ensemble is 0)
	while(!kbhit())
	{
		if(choice_lmode == '2')
		{
			hidden_max= output_max = 0;
			hidden_min = output_min = 9999;
			for(y=DIFF_DATA_SIZE; y<number_of_input_patterns+DIFF_DATA_SIZE; y++) {
				    ens_forward_pass(0,y);
				    ens_backward_pass(0,y,0.0);
			}
			hidden_range = hidden_max - hidden_min;
			output_range = output_max - output_min;
		}
	
		if(compare_output_to_target_tbg(0)) {
			cout << endl << "training finished" << endl;
			cout << endl << "just press a key to continue to write learning error into file" << endl;
			char temp_char = getch();
			break;
		}
		iter++;
		//else
		//fprintf(learn_file,"%d    %f   %f   %f   %f\n",(int)iter, RMSE, MAE, CE, CD);
	}

	count=0;
	for(i= 0; i< number_of_input_patterns; i++)
	{   	
		if(output[0][i+DIFF_DATA_SIZE] >= CLASS_THRESHOLD)
			pred[i] = 1.0;
		else 
			pred[i] = 0.0;
		//pred[i] = (int)((goutput[i]-outoffset)/outscale+0.5); 
		obs[i] = target[i+DIFF_DATA_SIZE];  //(int)((target[i]-outoffset)/outscale);
		if(pred[i] == obs[i])
		{
			temp = 1;
			count++;
		}
		else
			temp = 0;
	}
	accuracy[0]=(double)count/(double)number_of_input_patterns;
	printf("TrBagg - 0th NN on target:accuracy of classification is %f  %% \n", accuracy[0]*100.0);

	//select other NNs based on accuracy over same
	double pred1[SAME_TRAIN_DATA_SIZE], obs1[SAME_TRAIN_DATA_SIZE];
	count =0;
   	number_of_input_patterns = SAME_TRAIN_DATA_SIZE;

	/*delete []hidden;
	hidden_array_size = HD_ARRAY_SIZE;

	hidden = new double *[ensn+3];
	for(x=0; x< ensn+3; x++)
	{
		hidden[x] = new double [hidden_array_size];
		if(!hidden[x]) { cout << endl << "memory problem!"; exit(1); }
	}*/

	for(pattern = DIFF_DATA_SIZE; pattern< DIFF_DATA_SIZE+number_of_input_patterns; pattern++)
	{
		 for(tbgi=1; tbgi<ensn; tbgi++)
           ens_forward_pass(tbgi,pattern);
	}

	for(tbgi=1; tbgi<ensn; tbgi++)
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
			pred1[x] = -9999;
			obs1[x] = -100000;
			if(output[tbgi][x+DIFF_DATA_SIZE] >= CLASS_THRESHOLD)
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
		accuracy[tbgi]=(double)count/(double)number_of_input_patterns;
		printf("TrBagg-(target data only) validation: %d th NN - accuracy of classification is %f  %% \n", tbgi, accuracy[tbgi]*100.0);
		fclose(outfile1);
	}

	double tempmax; int temppos, temp1;
	ens_mark[0] =0;

	for(i =1; i< ensn; i++)
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

	bagg_mark =0; 
	for(i=0; i<ensn; i++)
	{
		if(accuracy[ens_mark[i]] >= accuracy[0])
                 bagg_mark++;
	} 
}

void bagg_forward_pass(int i,int pattern)
{
	register double temp=0;
	register int x,y;


	// INPUT -> HIDDEN
	for(y=0; y<hidden_array_size; y++) {
		for(x=0; x<input_array_size; x++) {
			temp += (bagg_input[i][pattern][x] * weight_i_h[i][x][y]);
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



void bagg_backward_pass(int i,int pattern)
{
	register int x, y;
	register double temp = 0;

	// COMPUTE ERRORSIGNAL FOR OUTPUT UNITS
	//errorsignal_output[x] = (target[pattern][x] - output[pattern][x]);  //original
	errorsignal_output[i] = output[i][pattern] * (1-output[i][pattern]) * (bagg_target[i][pattern]-output[i][pattern]); //NCL version
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
		length += bagg_input[i][pattern][x]*bagg_input[i][pattern][x];
	}
	if (length<=0.1) length = 0.1;
	for(x=0; x<input_array_size; x++) {
		for(y=0; y<hidden_array_size; y++) {
			//weight_i_h[x][y] += (learning_rate * errorsignal_hidden[y]*input[pattern][x]/length); //original 
			weight_i_h[i][x][y]+=learning_rate*errorsignal_hidden[i][y] * 
				bagg_input[i][pattern][x];
		}
	}

	// ADJUST BIASES FOR OUTPUT UNITS
	for(x=0; x<hidden_array_size; x++) {
		//bias[x] += (learning_rate * errorsignal_hidden[x] / length); //original
		bias[i][x] += learning_rate * errorsignal_hidden[i][x]; 
	}

	return;
}

int compare_output_to_target_tbg(int i)
{
	register int y,ytemp=0,count=0;
	register double temp;
	MSE = 0.0;
	RMSE = 0.0;
	MAE = 0.0;
	double pred[BAGG_SIZE_MEM], obs[BAGG_SIZE_MEM];
	
	for(y= 0 ; y < number_of_input_patterns; y++) {
		if(i!=0)
            temp = output[i][y];
		else
			temp = output[i][y+DIFF_DATA_SIZE];

		if(temp >= CLASS_THRESHOLD)
			pred[y] = 1.0;
		else
			pred[y] = 0.0;
		if(i!=0)
		  obs[y] = bagg_target[i][y];  //(int)((target[y][0]-outoffset)/outscale); 
		else 
          obs[y] = target[y+DIFF_DATA_SIZE];
		if(obs[y] == pred[y])
		{
			temp = 1;count++;
		}
		else temp=0;
	}
	printf("output: min  %f  max  %f range  %f \n", output_min, output_max, output_range);
	printf("hidden: min  %f  max  %f range  %f \n", hidden_min, hidden_max, hidden_range);
	printf("TrBagg training: %d th NN - error rate is %f %% \n", i, 100*(1-(double)count/(double)number_of_input_patterns));
	printf("iter: %d \n", iter);
	printf("_____________________________________\n");
	
	if(iter >= MAX_TBG_ITER)
		return 1;
	else
	   return 0;
}