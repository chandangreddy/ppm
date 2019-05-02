double D[512][1024][1024];
double C[512][1024][1024];
double  A[512][1024][1024];
double  B[512][1024][1024];
double alpha, beta;

void bgemm() {

#pragma scop
for (int b = 0; b < 512; b++)
  for (int i = 0; i < 1024; i++)
    for (int j = 0; j < 1024; j++) 
        for (int k = 0; k < 1024; k++) 
            D[b][i][j] += A[b][i][k] * B[b][k][j];        
#pragma endscop
}
