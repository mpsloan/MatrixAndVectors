// main program to perform the matrix vector multiplication in parallel
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>
#include <math.h>

float dot(float *v1, float *v2, int N) {
  int i;
  float sum=0.0;
  for (i = 0; i < N; i++) {
    sum += v1[i]*v2[i];
  }
  return sum;
}


int main(int argc, char** argv) {

  // Standard MPI initialization calls
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int rank, ncpu, name_len;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ncpu);

  // Inputs
  int L1     = atoi(argv[1]);
  int L2     = atoi(argv[2]);
  int A      = atoi(argv[3]);
  int B      = atoi(argv[4]);
  int Ap     = atoi(argv[5]);
  int Bp     = atoi(argv[6]);

  // Constants
  float dx, dy, pi, norm1, norm2;
  pi=acos(-1.0);
  dx=2.0*pi/L1;
  dy=2.0*pi/L2;
  norm1=sqrt(2.0/L1);
  norm2=sqrt(2.0/L2);

  /* Calculates which processor has what part of the global elements
     when dealing with L1 vectors (code shown in class) */
  int Nr1,R1;
  float xlast;
  Nr1 = (int)(L1/ncpu); // All processors have at least this many elements
  R1 = (L1%ncpu);       // Remainder of above calculation
  if (rank<R1) {
    ++Nr1;             // First R processors have one more element
    xlast=-dx/2.0+rank*Nr1*dx;
  }else{
    xlast=-dx/2.0+R1*(Nr1+1)*dx+(rank-R1)*Nr1*dx;
  }
  
  /* Calculates which processor has what part of the global elements         
     when dealing with L2 vectors */
  int Nr2, R2, row;
  float ylast;
  Nr2 = (int)(L2/ncpu); 
  R2 = (L2%ncpu);      
  if (rank<R2) {
    ++Nr2;
    row = 1 + Nr2 * rank;
    ylast=-dy/2.0+rank*Nr2*dy;
  }else{
    row = 1 + (Nr2+1) * R2 + Nr2 * (rank - R2);
    ylast=-dy/2.0+R2*(Nr2+1)*dy+(rank-R1)*Nr1*dy;
  }

  // initializing vectors on partition
  float *V1 = (float *)malloc(sizeof(float) * Nr1);
  assert(V1 != NULL);
  float *V2 = (float *)malloc(sizeof(float) * Nr1);
  assert(V2 != NULL);
  float *V3 = (float *)malloc(sizeof(float) * Nr2);
  assert(V3 != NULL);
  float *Matrix[Nr2];
  assert(Matrix != NULL);

  int i, j;
  float Xi, Yi, a, b;
  Xi = xlast;
  Yi = ylast;  
  a = pi * Ap/180;
  b = pi * Bp/180;

  // for loop to fill in v1 and v2 partions
  for (i = 0; i < Nr1; i++) {
    Xi = Xi + dx;
    V1[i]=norm1*cos(A*Xi+a);
    V2[i]=norm1*sin(B*Xi+b);
  }

  // for loop to fill in v3 partition and instantiate Matrix rows
  for (i  = 0; i < Nr2; i++) {
    Yi = Yi + dy;
    V3[i]=norm2*cos(B*Yi+b);
    Matrix[i] = (float *)malloc(sizeof(float) * L1);
    assert(Matrix[i] != NULL);
  }

  // received data vectors for Nr1 and Nr2 (shown in class)
  int *recvcounts = NULL;
  if (rank == 0) {
    recvcounts = malloc(ncpu * sizeof(int));
  }

  int *recvcounts2 = NULL;
  if (rank == 0) {
    recvcounts2 = malloc(ncpu * sizeof(int));
  }

  // gathering Nr1 and Nr2 to put into respective recvcounts
  MPI_Gather(&Nr1, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&Nr2, 1, MPI_INT, recvcounts2, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // displacements to deal with global interactions for both Nr values
  int *displs = NULL;
  if (rank == 0) {
    displs = malloc(ncpu * sizeof(int));
    displs[0] = 0;
    for (i = 1; i < ncpu; i++) {
      displs[i] = displs[i-1] + recvcounts[i-1];
    }
  }
  
  int *displs2 = NULL;
  if (rank == 0) {
    displs2 = malloc(ncpu * sizeof(int));
    displs2[0] = 0;
    for (i = 1; i < ncpu; i++) {
      displs2[i] = displs2[i-1] + recvcounts2[i-1];
    }
  }
  
  // global vector 1 to multiply with V3
  float *VOne = NULL;
  VOne = (float *)malloc(sizeof(float) * L1);

  
  // filling in V1 same way I did on scalar
  for (i = 0; i < L1; i++) {
    Xi = (pi/L1) + (i*dx);
    VOne[i]=norm1*cos(A*Xi+a);
  }

  /* global index to keep track of how much of the matrix the other
     cores cycled through before this rank */
  int globalIndex;
  float diagonal = 0.0;
  float nonDiagonal = 0.0;
  
  // setting initial global index to 0
  if (rank == 0) {
    globalIndex = 1;
  }
  else { // if you are not rank 0, receive the global index from the rank before you
    MPI_Recv(&globalIndex, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // filling in matrix partitions
  for (i = 0; i < Nr2; i++) {
    for (j = 0; j < L1; j++) {
      // part of v3 * entire global vector one
      Matrix[i][j] = V3[i] * VOne[j];
    }
  }

  // add up the diagonal and non-diagonal
  for (i = 0; i < Nr2; i++) {
    for (j = 0; j < L1; j++) {
      // find the diagonal element
      if (j == globalIndex - 1) {
        diagonal += Matrix[i][j];
      }
      else { // else it's a non-diagonal
        nonDiagonal += Matrix[i][j];
      }
    }
    // increment after every row
    ++globalIndex;
  }

  // as long as you aren't the last rank, send globalIndex to the rank after you
  if (rank != ncpu - 1) {
    MPI_Send(&globalIndex, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
  }
  
  // get local sum and then combine all cores to form global
  float localSum, globalSum, totalDiagonal, totalNonDiagonal;
  localSum = dot(V1, V3, Nr2);
  MPI_Reduce(&localSum, &globalSum, 1, MPI_FLOAT, MPI_SUM, 0,
	     MPI_COMM_WORLD);

  // get all the ranks diagonal and non-diagonal values
  MPI_Reduce(&diagonal, &totalDiagonal, 1, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&nonDiagonal, &totalNonDiagonal, 1, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // print the information
  if (rank == 0) {
    printf("The dot product of V1 * V3 is %f \n", globalSum);
    printf("V1 and V2 should be %d elements long. V3 should be %d elements long\n", L1, L2);
    printf("The total diagonal is: %f\n", totalDiagonal);
    printf("The total non-diagonal is: %f\n", totalNonDiagonal);
}

  // Clean up Memory
  free(V1);
  free(V2);
  free(V3);
  free(Matrix);
  free(VOne);
  free(displs);
  free(displs2);
  free(recvcounts);
  free(recvcounts2);

  MPI_Finalize();
}
