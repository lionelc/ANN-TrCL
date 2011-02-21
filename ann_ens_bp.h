//Backpropagation artificial neural network
//modified by Lei Jiang, Center for Computation and Technology, Louisiana State University

//the code is now for testbed only. The robustness of functionality is not guaranteed. The original author is glad to know any major modification of key functionality
//by the user of codes.

//added by LJ
#define TRAINING_DATA_SIZE  3566   //490  //6200 //363  //4000   //36   //814 //2575 //1600//1592
#define VALIDATION_DATA_SIZE  4558  //4558  //200  //4573  //1924  //814  

//mushroom  3516/100/4508/total:8124
//Australian   468/22/200/total:690       361/29/300


//add Jan 25 for transfer learning
#define DIFF_DATA_SIZE  3516    //361
#define SAME_TRAIN_DATA_SIZE 50 //50  //300   //29  //60  //35 //1300
#define SAME_TEST_DATA_SIZE  4558    //4558  //269   //4568   //4573

#define MAX_OBS_VALUE    600
#define MIN_OBS_VALUE    0

//added Jan 23 for ann ensemble
#define NUM_ENS      7      //7  //9 //7  //21  //11   //8
#define NCL_LAMBDA   0.703125  //0.7   //0.5  //0.51  //0.51  //0.7 // 0.7  //1.0  //0.703125
#define MAX_ITER     150

//Apr 26 experiments:  6 combinations in ensemble_number/ncl_lambda
//     5/0.7    5/0.5    5/0.3    
//     7/0.7    7/0.5    7/0.3

//added Apr 26 for ROC curve
#define CLASS_THRESHOLD 0.5

#define MAX_LINE_LENGTH 256
#define LINE_THRESHOLD 10

#define IN_ARRAY_SIZE 21   //14  //21 //3  //8
#define HD_ARRAY_SIZE 8    //7     //6  //4
#define LEARN_RATE  0.05

#define SCALE_COFF  3 //1.5  //0.65   //2.5  //1.5         //1.55        //1.35 //0.45  //1.8
#define SCALE_COFF_VAL  35 //1.5  //0.65   //2.5   //1.5
#define MLLW   0.993   //0.45
#define VAL_OFFSET 0

#define PRED_PERIOD  1  //24   //24   //6
#define START_MARK   50

//added July 2009
#define OVERSCALE_THRES 115.0  //120.0
#define OVER_SCALE  60.0  //50

//added Sep 2009
#define CENTER_SIZE  20 //100  //100     //216  //216 //36   
#define PERTURBED_SEGMENTS  10//6  //6

//variables
double **input,
**hidden,
**output,
*target,
**bias,
***weight_i_h,
**weight_h_o,
**errorsignal_hidden,
*errorsignal_output;

//added Jan 23 for batch mode
double ***delta_weight_i_h,
**delta_weight_h_o, **delta_bias;
char choice_lmode='2';

int input_array_size,
hidden_array_size,
max_patterns,
bias_array_size,
gaset = -2500,
number_of_input_patterns,
pattern,
file_loaded = 0,
ytemp = 0,
ztemp = 0;
double learning_rate,
max_error_tollerance = 0.55;
char filename[128];

long int iter;
double learn_inc[6];
double temp_RMSE[6];

//model assessment variables
double MSE;   //mean squared error
double MSRE;  //mean squared relative error

double CE;    //coefficient of efficiency
double CD;    //coefficient of determination

double MAE;   //mean absolute error
double RMSE;  //root mean squared error

double maxCE;  //current maximum CE to determine when to stop learning process

//added by Jul 3
double AARE;

//added by Nov 11  -- new metrics
double RE;

//added Sep 30, 2009
double omega_c[TRAINING_DATA_SIZE];
double radius_c[TRAINING_DATA_SIZE];
double size_c[TRAINING_DATA_SIZE];
double maxw_c[TRAINING_DATA_SIZE];
double center_weight[3];
double wave_scale;

//added Jan 22 for letter datasets
void init_data_memory();
int    **input_letter;
int    *letter_class;

//added Jan 23 for mushroom datasets
int   **input_mushroom;
int    *mushroom_class;

//added Jan 27 for Australian datasets
double **input_aus;
int *aus_class;

double hidden_min, hidden_max, hidden_range;
double output_min, output_max, output_range;

//added Jan 24 ensemble number
int ensn = NUM_ENS;
double goutput[TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE];
int ens_mark[NUM_ENS+3];  //used for selection  //size changed Apr 29

double max_wave_sim[TRAINING_DATA_SIZE][4];

//added Oct 27, 2009
double toy_input[TRAINING_DATA_SIZE];
double toy_output[TRAINING_DATA_SIZE];

//added Dec 2, 2009
void scale_input_output_RBF();
double minin[IN_ARRAY_SIZE], maxin[IN_ARRAY_SIZE], minout, maxout;
double inscale[IN_ARRAY_SIZE], outscale, inoffset[IN_ARRAY_SIZE], outoffset;
int  inlist[IN_ARRAY_SIZE][21];

//functions
int compare_output_to_target(); //modified Apr 23
void load_data(char *arg);
void save_data(char *argres);
void ens_forward_pass(int i,int pattern);
void ens_backward_pass(int i,int pattern, double lambda);
void learn();
void change_learning_rate();
void initialize_net();
void clear_memory();

void Initialization(int pattern_count);
void init_input_output();
void validate();
void read_and_go(char* infile1, char* infile2);
void synthesis();  //added May 21
void validate_input_output();  //add Jan 23, 2010

//added Jul 20
void basic_memory_assign();

//added Sep 30
void init_cweights();
void init_center_RBF();
void scale_center_RBF();

//add Jan 22
void perturb_training_data();
void scale_input_output();

//add Jan 23 for batch mode learning
void ens_backward_pass_batch(int i, int pattern);


//subroutines for calculation
double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-1.0 * x));
}

double rev_sigmoid(double x)
{
	return log(1.0 / (1.0/x -1));
}

//model assessment
void calMSE(double pred[], double obs[], int size, double result)
{
	int i, miss_value = 0;
    double error = 0.0;

	for(i=0; i< size; i++)
	{
	  if(obs[i] < MAX_OBS_VALUE)
	    error+= (pred[i]-obs[i])*(pred[i]-obs[i]);
	  else
		miss_value++;
	}

	result = error/(size-miss_value);
}

void calRMSE(double pred[], double obs[], int size, double& result)
{
	int i, miss_value =0;
    double error = 0.0;
	
	for(i=0; i< size; i++)
	{
		if(obs[i] < MAX_OBS_VALUE)
		    error+= (pred[i]-obs[i])*(pred[i]-obs[i]);
		else
			miss_value++;
	}
	
	result = sqrt(error/(size-miss_value));
}

void calMAE(double pred[], double obs[], int size, double& result)
{
	int i, miss_value = 0;
    double error = 0.0;
	
	for(i=0; i< size; i++)
	{
       if(obs[i] < MAX_OBS_VALUE)
		  error+= fabs(pred[i]-obs[i]);
	   else
		   miss_value++;
	}
	
	result = error/(size-miss_value);
}

void calAARE(double pred[], double obs[], int size, double& result)
{
	int i, miss_value = 0;
    double error=0;
	
	for(i=0; i< size; i++)
	{
		if(obs[i] < MAX_OBS_VALUE)
		{
			error+= fabs(pred[i]-obs[i])/obs[i];
		}
		else
			miss_value++;
	}
	
	result = error/(size-miss_value);
}

void calMSRE(double pred[], double obs[], int size, double &result)
{
	int i, miss_value =0;
    double error = 0.0;
	double suberror=0.0;
	
	for(i=0; i< size; i++)
	{
		if(obs[i] < MAX_OBS_VALUE)
		{
			suberror = (pred[i]-obs[i])*(pred[i]-obs[i]);
			suberror = suberror / (obs[i]*obs[i]);
			error += suberror;
		}
		else
			miss_value++;
	}
	
	result = error/(size-miss_value);
}

void calCE(double pred[], double obs[], int size, double &result)
{
	int i;
	double meanobs=0.0, error1 =0.0, error2 =0.0;

	for(i=0; i< size; i++)
	{
		if(obs[i] < MAX_OBS_VALUE)
           meanobs+=obs[i];
	}
	meanobs/= size;

	for(i=0; i<size; i++)
	   if(obs[i] < MAX_OBS_VALUE)
	   {
        error1+= (obs[i]-pred[i])*(obs[i]-pred[i]);
		error2+= (obs[i]-meanobs)*(obs[i]-meanobs);
	   }

	result = 1- error1/error2;
}

/*void calCD(double pred[], double obs[], int size, double &result)
{
	int i;
	double meanobs=0.0, meanpred=0.0, error1 =0.0, error2 =0.0, error3 = 0.0;
	
	for(i=0; i< size; i++)
	{
		if(obs[i] < MAX_OBS_VALUE)
		{
           meanobs+=obs[i];
		   meanpred+=pred[i];
		}
	}
	meanobs/= size;
	meanpred/= size;
	
	for(i=0; i<size; i++)
	{
        if(obs[i] < MAX_OBS_VALUE)
		{
			error1+= (pred[i]-meanpred)*(obs[i]-meanobs);
		    error2+= (pred[i]-meanpred)*(pred[i]-meanpred);
		    error3+= (obs[i]-meanobs)*(obs[i]-meanobs);
		}
	}
	
	result = (error1/sqrt(error2*error3))*(error1/sqrt(error2*error3));
}*/

void calCD(double pred[], double obs[], int size, double &result)
{
	int i;
	double meanobs=0.0, meanpred=0.0, error1 =0.0, error2 =0.0;
	
	for(i=0; i< size; i++)
	{
		if(obs[i] < MAX_OBS_VALUE)
		{
           meanobs+=obs[i];
		   meanpred+=pred[i];
		}
	}
	meanobs/= size;
	meanpred/= size;
	
	for(i=0; i<size; i++)
	{
        if(obs[i] < MAX_OBS_VALUE)
		{
			error1+= (pred[i]-obs[i])*(pred[i]-obs[i]);
		    error2+= (obs[i]-meanobs)*(obs[i]-meanobs);
		}
	}
	
	result = 1-error1/error2;
}

//new metrics -- added by 11/11/2009
void calRE(double pred[], double obs[], int size, double& result)
{
	int i, miss_value = 0;
    double error=0, error2=0;;
	
	for(i=0; i< size; i++)
	{
		if(obs[i] < MAX_OBS_VALUE)
		{
			error+= (pred[i]-obs[i])*(pred[i]-obs[i]);
			error2+= obs[i]*obs[i];
		}
		else
			miss_value++;
	}
	
	result = sqrt(error/error2);
}



//local statistics
double max(double* array, int start, int end)
{
	double maxv = array[start];
	for(int i=start+1; i<= end; i++)
		if(maxv < array[i])
			maxv = array[i];
		
		return maxv;
}

double mean(double* array, int start, int end)
{
	double sum = 0.0, meanv;
	for(int i=start; i<= end; i++)
		sum+=array[i];
	
	meanv = sum/(end-start+1);
    return meanv;
}

double Limiter(double value)
{
	if(value > MAX_OBS_VALUE) value = MAX_OBS_VALUE;
	if(value < MIN_OBS_VALUE) value = MIN_OBS_VALUE;
	return value;
}

int Integer(double value)
{
	if(value-(int)value >= 0.5) return (int)value+1;
		else return (int)value;
}