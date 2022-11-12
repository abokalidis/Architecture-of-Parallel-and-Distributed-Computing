#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <float.h>
#include <xmmintrin.h>
#include <pthread.h>
#include <unistd.h>
#include <mpi.h>


#define MINSNPS_B 5
#define MAXSNPS_E 20
#define EXIT 127
#define BUSYWAIT 0
#define INITIALIZE 125
#define COMPUTE 126

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MAX4(a,b,c,d) MAX(0, MAX(a,MAX(b,MAX(c,d))))

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MIN4(a,b,c,d) MIN(0, MIN(a,MIN(b,MIN(c,d))))


typedef struct
{
    __m128 *LVec128;
    __m128 *RVec128;
    __m128 *CVec128;
    __m128 *FVec128;
    __m128 *mVec128;
    __m128 *nVec128; 
    float maxF;
    float minF;
    float avgF;
    int world_rank;
} data_t;

typedef struct 
{
    int threadID;
    int threadTOTAL;

    int threadBARRIER;
    int threadOPERATION;

    int threadSTART;
    int threadEND;

    data_t dt;    

} threadData_t;


double gettime(void) {
    struct timeval ttime;
    gettimeofday(&ttime , NULL);
    return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

float randpval (void) {
    int vr = rand();
    int vm = rand()%vr;
    float r = ((float)vm)/(float)vr;
    assert(r>=0.0f && r<=1.00001f);
    return r;
}

void initializeThreadData(threadData_t * cur, int i, int threads, float * mVec, float * nVec, float * LVec, float * RVec, float * CVec, float * FVec,int world_rank)
{
    cur->threadID = i;
    cur->threadTOTAL = threads;
    cur->threadBARRIER = 0;
    cur->threadOPERATION = BUSYWAIT;
    cur->threadSTART = 0;
    cur->threadEND = 0;
    cur->dt.mVec128 = (__m128 *) mVec;
    cur->dt.nVec128 = (__m128 *) nVec;
    cur->dt.LVec128 = (__m128 *) LVec;
    cur->dt.RVec128 = (__m128 *) RVec;
    cur->dt.CVec128 = (__m128 *) CVec;
    cur->dt.FVec128 = (__m128 *) FVec;
    cur->dt.avgF = 0.0f;
    cur->dt.maxF = 0.0f;
    cur->dt.minF = FLT_MAX;
    cur->dt.world_rank = world_rank;
}

void setThreadOperation(threadData_t *threadData, int operation){

    int i, threads = threadData[0].threadTOTAL;

    for(i=0;i<threads;i++)
        threadData[i].threadOPERATION = operation;
}

void *initializeSSE(threadData_t *threadData){
}

void syncThreadsBARRIER(threadData_t * threadData)
{
    int i, threads = threadData[0].threadTOTAL, barrierS=0;

    threadData[0].threadOPERATION=BUSYWAIT;


    while(barrierS!=threads)
    {
        barrierS=0;
        for(i=0;i<threads;i++)
            barrierS += threadData[i].threadBARRIER;
    }

    for(i=0;i<threads;i++)
        threadData[i].threadBARRIER=0;
}



void * computeDATA (threadData_t * threadData)
{
    float avgF = 0.0f;
    float maxF = 0.0f;
    float minF = FLT_MAX;
    __m128 comparemaxFlags;
    __m128 compareminFlags;
    __m128 minFVec128 = _mm_set_ps1(FLT_MAX);
    __m128 avgFVec128 = _mm_set_ps1(0);
    __m128 maxFVec128 = _mm_set_ps1(FLT_MIN);

    
    __m128 vec001 = _mm_set1_ps(0.01f); 
    __m128 vec1 = _mm_set1_ps(1.f);
    __m128 vec2 = _mm_set1_ps(2.f); 


//    printf("world_rank:%d\n", threadData->dt.world_rank);
//    printf("tdi:%d\n", threadData->threadID);


    for(unsigned int i=threadData->threadSTART;i<threadData->threadEND;i++)
    {
        __m128 num_0 = _mm_add_ps( threadData->dt.LVec128[i], threadData->dt.RVec128[i]);
        __m128 num_1 = _mm_div_ps( _mm_mul_ps( threadData->dt.mVec128[i], _mm_sub_ps( threadData->dt.mVec128[i], vec1)), vec2);
        __m128 num_2 = _mm_div_ps( _mm_mul_ps( threadData->dt.nVec128[i], _mm_sub_ps( threadData->dt.nVec128[i], vec1)), vec2);
        __m128 num = _mm_div_ps( num_0, _mm_add_ps( num_1, num_2));
        __m128 den_0 = _mm_sub_ps( _mm_sub_ps( threadData->dt.CVec128[i], threadData->dt.LVec128[i]), threadData->dt.RVec128[i]);
        __m128 den_1 = _mm_mul_ps( threadData->dt.mVec128[i], threadData->dt.nVec128[i]);
        __m128 den = _mm_div_ps( den_0, den_1);
        threadData->dt.FVec128[i] = _mm_div_ps(num, _mm_add_ps(den, vec001));
        comparemaxFlags = _mm_cmpgt_ps( threadData->dt.FVec128[i], maxFVec128);
        compareminFlags=  _mm_cmplt_ps( threadData->dt.FVec128[i], minFVec128);
       
        maxFVec128 = _mm_or_ps(_mm_and_ps( comparemaxFlags, threadData->dt.FVec128[i]),  _mm_andnot_ps(comparemaxFlags, maxFVec128));
        minFVec128 = _mm_or_ps(_mm_and_ps( compareminFlags, threadData->dt.FVec128[i]),  _mm_andnot_ps(compareminFlags, minFVec128));
        avgFVec128 = _mm_add_ps(avgFVec128, threadData->dt.FVec128[i]);
        
    }

    __m128 tempMax = _mm_max_ps(maxFVec128, _mm_shuffle_ps(maxFVec128, maxFVec128, _MM_SHUFFLE(0,0,3,2)));
    maxF = _mm_cvtss_f32(_mm_max_ps(tempMax, _mm_shuffle_ps(tempMax, tempMax, _MM_SHUFFLE(0,0,0,1))));
    __m128 tempMin = _mm_min_ps(minFVec128, _mm_shuffle_ps(minFVec128, minFVec128, _MM_SHUFFLE(0,0,3,2)));
    minF = _mm_cvtss_f32(_mm_min_ps(tempMin, _mm_shuffle_ps(tempMin, tempMin, _MM_SHUFFLE(0,0,0,1))));
    __m128 tempAvg = _mm_add_ps(avgFVec128, _mm_shuffle_ps(avgFVec128, avgFVec128, _MM_SHUFFLE(0,0,3,2)));
    avgF = _mm_cvtss_f32(_mm_add_ps(tempAvg, _mm_shuffle_ps(tempAvg, tempAvg, _MM_SHUFFLE(0,0,0,1))));

    threadData->dt.avgF = avgF;
    threadData->dt.maxF = threadData->dt.maxF>=maxF?threadData->dt.maxF:maxF;
    threadData->dt.minF = threadData->dt.minF<=minF?threadData->dt.minF:minF;
}



void startThreadOperations(threadData_t * threadData, int operation)
{
    setThreadOperation(threadData, operation);
    computeDATA(threadData);
    threadData[0].threadBARRIER=1;
    syncThreadsBARRIER(threadData);     
}

void * threadFUNC (void * x)
{
    threadData_t * currentThread = (threadData_t *)x;

    while(1) {
        __sync_synchronize();

        if (currentThread->threadOPERATION == EXIT)
            return NULL;

        if(currentThread->threadID == 0 && currentThread->threadOPERATION == INITIALIZE){
            initializeSSE(currentThread);
            currentThread->threadOPERATION = BUSYWAIT;
        }

        if (currentThread->threadOPERATION == COMPUTE) {
            computeDATA(currentThread);
            currentThread->threadOPERATION = BUSYWAIT;
            currentThread->threadBARRIER = 1;
            while (currentThread->threadBARRIER == 1) __sync_synchronize();
        }
    }
}

void terminateWorkerThreads(pthread_t * workerThreadL, threadData_t * threadData)
{
    int i, threads=threadData[0].threadTOTAL;
    
    for(i=0;i<threads;i++)
        threadData[i].threadOPERATION = EXIT;           

    for(i=1;i<threads;i++)
        pthread_join(workerThreadL[i-1],NULL);
}



int main(int argc, char ** argv) {
	assert(argc == 3);

    double timeTotalMainStart = gettime();

    unsigned int N = (unsigned int) atoi(argv[1]);
    unsigned int threads = (unsigned int) atoi(argv[2]);
    unsigned int iters = 1;
    float minF=FLT_MAX;
    float maxF=0.0f;
    float avgF=0.0f;
    float *max, *min  = NULL;

    int world_size;
    int world_rank;
    MPI_Init(NULL, NULL);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    srand(1);

    float * mVec = (float*)malloc(sizeof(float)*N);
    assert(mVec != NULL);

    float * nVec = (float*)malloc(sizeof(float)*N);
    assert(nVec != NULL);

    float * LVec = (float*)malloc(sizeof(float)*N);
    assert(LVec != NULL);

    float * RVec = (float*)malloc(sizeof(float)*N);
    assert(RVec != NULL);

    float * CVec = (float*)malloc(sizeof(float)*N);
    assert(CVec != NULL);

    float * FVec = (float*)malloc(sizeof(float)*N);
    assert(FVec != NULL);



    for(unsigned int i=0;i<N;i++) {
        mVec[i] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
        nVec[i] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
        LVec[i] = randpval()*mVec[i];
        RVec[i] = randpval()*nVec[i];
        CVec[i] = randpval()*mVec[i]*nVec[i];
        FVec[i] = 0.0;
        assert(mVec[i]>=MINSNPS_B && mVec[i]<=(MINSNPS_B+MAXSNPS_E));
        assert(nVec[i]>=MINSNPS_B && nVec[i]<=(MINSNPS_B+MAXSNPS_E));
        assert(LVec[i]>=0.0f && LVec[i]<=1.0f*mVec[i]);
        assert(RVec[i]>=0.0f && RVec[i]<=1.0f*nVec[i]);
        assert(CVec[i]>=0.0f && CVec[i]<=1.0f*mVec[i]*nVec[i]);
    }

    double timeOmegaTotalStart = gettime();
  
    pthread_t *workerThread = (pthread_t *) malloc (sizeof(pthread_t)*((unsigned long)(threads-1)));
        
    threadData_t * threadData = (threadData_t *) malloc (sizeof(threadData_t)*((unsigned long)threads));
    assert(threadData!=NULL);

    for(int i=threads-1;i>=0;i--){

        initializeThreadData(&threadData[i],i,threads,mVec,nVec,LVec,RVec,CVec, FVec,world_rank);
	    if(i>=1)
            pthread_create (&workerThread[i-1], NULL, threadFUNC, (void *) (&threadData[i]));
	}


    for(int i=0; i<threads; i++){
            int start = ( (N/4) * i + (threads-1) ) / threads;
            int end  = ( (N/4) * (i+1) + (threads-1) ) / threads;
            threadData[i].threadSTART = start;
            threadData[i].threadEND = end;
    }


    startThreadOperations(threadData, COMPUTE);
    
    for(unsigned int i=0;i<threads;i++)
    {
        maxF = threadData[i].dt.maxF>maxF?threadData[i].dt.maxF:maxF;
        minF = threadData[i].dt.minF<minF?threadData[i].dt.minF:minF;
	    avgF+=threadData[i].dt.avgF;
    }

    if(N % 4 > 0){
        if(threadData->threadID == 0){
            int loopSTART = N - (N%4);
            int loopSTOP = N;
                for (int i = loopSTART; i < loopSTOP; i++) {
                    float num_0 = LVec[i]+RVec[i];
                    float num_1 = mVec[i]*(mVec[i]-1.0f)/2.0f;
                    float num_2 = nVec[i]*(nVec[i]-1.0f)/2.0f;
                    float num = num_0/(num_1+num_2);

                    float den_0 = CVec[i]-LVec[i]-RVec[i];
                    float den_1 = mVec[i]*nVec[i];
                    float den = den_0/den_1;
                    
                    FVec[i] = num/(den+0.01f);
                    
                    maxF = FVec[i]>maxF?FVec[i]:maxF;
                    minF = FVec[i]<minF?FVec[i]:minF;
                    avgF += FVec[i];
                }
            }
    }

    terminateWorkerThreads(workerThread,threadData);
 
    
    if (world_rank == 0) {
        min = (float *) malloc(world_size * sizeof(float));
        assert(min != NULL);
        max = (float *) malloc(world_size * sizeof(float));
        assert(max != NULL);
    }

    MPI_Gather(&minF, 1, MPI_FLOAT, min, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&maxF, 1, MPI_FLOAT, max, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // Sync and finalize the MPI environment. No more MPI calls can be made after this
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    double timeOmegaTotal = gettime() - timeOmegaTotalStart;
    double timeTotalMainStop = gettime();
    float fmaxF;
    float fminF;
    if (world_rank == 0) {
    	fmaxF = 0.0f;
    	fminF = FLT_MAX;
        int k=0;
        while(k<world_size){
            fmaxF = MAX(max[k], maxF);
            fminF = MIN(min[k], minF);
            k++;
        }
        printf("Omega time %fs\nTotal time %fs\nMin %e\nMax %e\nAvg %e\n", timeOmegaTotal/iters, timeTotalMainStop-timeTotalMainStart, (double)fminF, (double)fmaxF,(double)avgF/N);
    }

    

    
    free(mVec);
    free(nVec);
    free(LVec);
    free(RVec);
    free(CVec);
    free(max);
    free(min);
    exit(0);
}
