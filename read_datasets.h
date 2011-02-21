//read hurricane-related data into neural network
//written by Lei Jiang, Center for Computation and Technology, Louisiana State University

//the code is now for testbed only. The robustness of functionality is not guaranteed. The original author is glad to know any major modification of key functionality
//by the user of codes.

#define LEARNING_FILE_PATH_RBF  ".\\learning_error_archive\\learning_error_RBF.txt"
#define VALIDATION_OUT_FILE_PATH_RBF  ".\\learning_error_archive\\validation_error_RBF.txt"

#define LEARNING_FILE_PATH   ".\\learning_error_archive\\learning_error.txt"
#define VALIDATION_OUT_FILE_PATH1  ".\\learning_error_archive\\tf_validation_error_counter.txt"

#define WEIGHT_I_H_FILE ".\\learning_error_archive\\weight_i_h_counter.txt"
#define WEIGHT_H_O_FILE ".\\learning_error_archive\\weight_h_o_counter.txt"

#define WEIGHT_I_H_FILE_RBF ".\\learning_error_archive\\weight_i_h_rbf_counter.txt"
#define WEIGHT_H_O_FILE_RBF ".\\learning_error_archive\\weight_h_o_rbf_counter.txt"

#define TRAINING_FILE_PATH_TEST  ".\\learning_error_archive\\test.txt"
#define VALIDATION_FILE_PATH_TEST  ".\\learning_error_archive\\test.txt"

#define LEARNING_CURVE_PATH ".\\learning_error_archive\\learning_curve_counter.txt"


void read_output_file(char* in_filename, int start, int end, double* obs, double* pred)
{
	int i=0, j=0; 
	int count =0, timecount=0; //count the num of edges
	int firstmark =0;
	char tempstr[MAX_LINE_LENGTH];
	char  tempchar[LINE_THRESHOLD];
	for(i=0;i< MAX_LINE_LENGTH;i++)
		tempstr[i] = '\0';
	
	bool startmark = false;
	int startcount =0;
	
	FILE *file;
	file = fopen (in_filename, "r");
	if (!file)
	{
		printf ("Error opening %s\n", in_filename);
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
		
		if(!startmark)
		{
			if(startcount< start-1)
			{
				startcount++;
				continue;
			}
			else
				startmark = true; 
		}
		
		while(tempstr[i] != ' ' && tempstr[i] != 9)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}

		//added May 21
        timecount = atoi(tempchar);

		for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		
		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		obs[count] = atof(tempchar);
		
		for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		
		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		pred[count] = atof(tempchar);
		count++;
		
		if(count>= end-start+1)
			break;
	}
	
    fclose(file);
	return;
}

void read_max_wave_file(char* in_filename, int start, int end)
{
	int i=0, j=0; 
	int count =0; //count the num of edges
	int firstmark =0;
	char tempstr[MAX_LINE_LENGTH];
	char  tempchar[LINE_THRESHOLD];
	for(i=0;i< MAX_LINE_LENGTH;i++)
		tempstr[i] = '\0';
	
	bool startmark = false;
	int startcount =0;
	
	FILE *file;
	file = fopen (in_filename, "r");
	if (!file)
	{
		printf ("Error opening %s\n", in_filename);
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
		
		if(!startmark)
		{
			if(startcount< start-1)
			{
				startcount++;
				continue;
			}
			else
				startmark = true; 
		}
		
		while(tempstr[i] != ' ' && tempstr[i] != 9)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		omega_c[count] = atof(tempchar);
		
		for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		
		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		size_c[count] = atof(tempchar);

		for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		
		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		radius_c[count] = atof(tempchar);

		for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		
		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		maxw_c[count] = atof(tempchar);

		count++;
		
		if(count>= end-start+1)
			break;
	}
	
    fclose(file);
	return;
}

void read_cos_file(char* in_filename, int start, int end)
{
	int i=0, j=0; 
	int count =0; //count the num of edges
	int firstmark =0;
	char tempstr[MAX_LINE_LENGTH];
	char  tempchar[LINE_THRESHOLD];
	for(i=0;i< MAX_LINE_LENGTH;i++)
		tempstr[i] = '\0';
	
	bool startmark = false;
	int startcount =0;
	
	FILE *file;
	file = fopen (in_filename, "r");
	if (!file)
	{
		printf ("Error opening %s\n", in_filename);
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
		
		if(!startmark)
		{
			if(startcount< start-1)
			{
				startcount++;
				continue;
			}
			else
				startmark = true; 
		}
		
		while(tempstr[i] != ' ' && tempstr[i] != 9)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		toy_input[count] = atof(tempchar);
		
		for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		
		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		toy_output[count] = atof(tempchar);

		count++;
		
		if(count>= end-start+1)
			break;
	}
	
    fclose(file);
	return;
}

void read_unary_file(char* in_filename, int start, int end)
{
	int i=0, j=0; 
	int count =0; //count the num of edges
	int firstmark =0;
	char tempstr[MAX_LINE_LENGTH];
	char  tempchar[LINE_THRESHOLD];
	for(i=0;i< MAX_LINE_LENGTH;i++)
		tempstr[i] = '\0';
	
	bool startmark = false;
	int startcount =0;
	
	FILE *file;
	file = fopen (in_filename, "r");
	if (!file)
	{
		printf ("Error opening %s\n", in_filename);
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
		
		if(!startmark)
		{
			if(startcount< start-1)
			{
				startcount++;
				continue;
			}
			else
				startmark = true; 
		}

		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}

		while(tempstr[i] != ' ' && tempstr[i] != 9)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		omega_c[count] = atof(tempchar);

		for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		
		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		radius_c[count] = atof(tempchar);

		for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		
		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		size_c[count] = atof(tempchar);
		
		for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		
		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		max_wave_sim[count][0] = atof(tempchar);

		for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		
		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		max_wave_sim[count][1] = atof(tempchar);

		for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		
		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		max_wave_sim[count][2] = atof(tempchar);

		for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		
		while(tempstr[i] == ' ' || tempstr[i] == 9)
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10)
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		max_wave_sim[count][3] = atof(tempchar);

		count++;
		
		if(count>= end-start+1)
			break;
	}
	
    fclose(file);
	return;
}


void read_letter_file(char* in_filename, int start, int end)
{
	int i=0, j=0; 
	int count =0; //count the num of edges
	int firstmark =0;
	char tempstr[MAX_LINE_LENGTH];
	char  tempchar[LINE_THRESHOLD];
	for(i=0;i< MAX_LINE_LENGTH;i++)
		tempstr[i] = '\0';
	
	bool startmark = false;
	int startcount =0;
	
	FILE *file;
	file = fopen (in_filename, "r");
	if (!file)
	{
		printf ("Error opening %s\n", in_filename);
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
		
		if(!startmark)
		{
			if(startcount< start-1)
			{
				startcount++;
				continue;
			}
			else
				startmark = true; 
		}
		
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i]!=',')
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		letter_class[count] = (int)tempchar[0]-65;
		
		for(int s=0; s<= 15; s++)
		{
			for(j=0; j< firstmark; j++)
			tempchar[j] = '\0';
		firstmark = 0;
		while(tempstr[i] == ' ' || tempstr[i] == 9 ||tempstr[i]==',')
		{
			i++;
		}
        
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10 && tempstr[i]!=',')
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		input_letter[count][s] = atoi(tempchar);
		}

		count++;
		
		if(count>= end-start+1)
			break;
	}
	
    fclose(file);
	return;
}

void read_mushroom_file(char* in_filename, int start, int end)
{
	int i=0, j=0; 
	int count =0; //count the num of edges
	int firstmark =0;
	char tempstr[MAX_LINE_LENGTH];
	char  tempchar[LINE_THRESHOLD];
	for(i=0;i< MAX_LINE_LENGTH;i++)
		tempstr[i] = '\0';
	
	bool startmark = false;
	int startcount =0;
	
	FILE *file;
	file = fopen (in_filename, "r");
	if (!file)
	{
		printf ("Error opening %s\n", in_filename);
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
		
		if(!startmark)
		{
			if(startcount< start-1)
			{
				startcount++;
				continue;
			}
			else
				startmark = true; 
		}
		
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i]!=',')
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		if(tempchar[0] == 'p')
	 	     mushroom_class[count] = 0;
		else if (tempchar[0] == 'e')
			 mushroom_class[count] = 1;
		
		for(int s=0; s< 10; s++)
		{
			for(j=0; j< firstmark; j++)
				tempchar[j] = '\0';
			firstmark = 0;
			while(tempstr[i] == ' ' || tempstr[i] == 9 ||tempstr[i]==',')
			{
				i++;
			}

			while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10 && tempstr[i]!=',')
			{
				tempchar[firstmark] = tempstr[i];
				i++;
				firstmark++;
			}

			input_mushroom[count][s] = (int)tempchar[0]-97;
		}
        	for(j=0; j< firstmark; j++)
				tempchar[j] = '\0';
			firstmark = 0;
			while(tempstr[i] == ' ' || tempstr[i] == 9 ||tempstr[i]==',')
			{
				i++;
			}

			while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10 && tempstr[i]!=',')
			{
				tempchar[firstmark] = tempstr[i];
				i++;
				firstmark++;
			}
         for(int s=10; s< 21; s++)
		{
			for(j=0; j< firstmark; j++)
				tempchar[j] = '\0';
			firstmark = 0;
			while(tempstr[i] == ' ' || tempstr[i] == 9 ||tempstr[i]==',')
			{
				i++;
			}

			while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10 && tempstr[i]!=',')
			{
				tempchar[firstmark] = tempstr[i];
				i++;
				firstmark++;
			}

			input_mushroom[count][s] = (int)tempchar[0]-97;
		}

		count++;
		
		if(count>= end-start+1)
			break;
	}
	
    fclose(file);
	return;
}

void read_aus_file(char* in_filename, int start, int end)
{
	int i=0, j=0, temp; 
	int count =0; //count the num of edges
	int firstmark =0;
	char tempstr[MAX_LINE_LENGTH];
	char  tempchar[LINE_THRESHOLD];
	for(i=0;i< MAX_LINE_LENGTH;i++)
		tempstr[i] = '\0';
	
	bool startmark = false;
	int startcount =0;
	
	FILE *file;
	file = fopen (in_filename, "r");
	if (!file)
	{
		printf ("Error opening %s\n", in_filename);
		exit(0);
	}
	
	while(fgets(tempstr,MAX_LINE_LENGTH,file))
	{
		i = 0; temp=0;
		firstmark = 0;
		
		for(j=0; j< LINE_THRESHOLD; j++)
			tempchar[j] = '\0';
		
		if(tempstr[0] == '"' || tempstr[0] == '#')
			continue;
		
		if(!startmark)
		{
			if(startcount< start-1)
			{
				startcount++;
				continue;
			}
			else
				startmark = true; 
		}
		
		while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i]!=',')
		{
			tempchar[firstmark] = tempstr[i];
			i++;
			firstmark++;
		}
		
		if(tempchar[0] == '-')
	 	     aus_class[count] = 0;
		else if (tempchar[0] == '+')
			 aus_class[count] = 1;

		for(int s=0; s<14 ;s++)
			input_aus[count][s] = 0.0;
		
		for(int s=0; s< 14; s++)
		{
			if(tempstr[i] == '\0' || tempstr[i] == 9 || tempstr[i]==10)
				break;

			for(j=0; j< firstmark; j++)
				tempchar[j] = '\0';
			firstmark = 0;
			while(tempstr[i] == ' '|| tempstr[i] == 9 ||tempstr[i]==',')
			{
				i++;
			}

			while(tempstr[i] != ' ' && tempstr[i]!=':' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10 )
			{
				tempchar[firstmark] = tempstr[i];
				i++;
				firstmark++;
			}
			temp = atoi(tempchar);
			if(temp <1)
				break;

			for(j=0; j< firstmark; j++)
				tempchar[j] = '\0';
			firstmark=0;

			i++;  //for the colon

			while(tempstr[i] != ' ' && tempstr[i] != 9 && tempstr[i] != '\0' && tempstr[i] != 10 && tempstr[i]!=',' && tempstr[i]!=':')
			{
				tempchar[firstmark] = tempstr[i];
				i++;
				firstmark++;
			}

			input_aus[count][temp-1] = atof(tempchar);

			if(temp == 14)
				break;
		}
        	
		count++;
		
		if(count>= end-start+1)
			break;
	}
	
    fclose(file);
	return;
}

