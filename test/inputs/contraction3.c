int C[1024][1024][1024];
int A[1024][1024];
int B[1024][1024][1024];

void contraction3() {

#pragma scop
  for (int i = 0; i < 1024; i++)
    for (int j = 0; j < 1024; j++) 
        for (int k = 0; k < 1024; k++)
          for (int l = 0; l < 1024; l++)
            C[i][j][k] += A[i][l] * B[j][l][k];        
#pragma endscop
}