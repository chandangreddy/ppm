double D[1024][1024];
double C[1024][1024];
double A[1024][1024];
double B[1024][1024];
double alpha, beta;

void tgemm() {

for (int i = 0; i < 1024; i++)
    for (int j = 0; j < 1024; j++) 
        B[i][j] = B[j][i];


#pragma scop
 for (int i = 0; i < 1024; i++) {
    for (int j = 0; j < 1024; j++) {
      for (int k = 0; k < 1024; k++) { 
	D[i][j] += alpha * A[i][k] * B[j][k];
      }
    }
 }
#pragma endscop
}
