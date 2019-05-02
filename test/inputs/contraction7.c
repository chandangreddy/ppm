int E[1024][1024][1024][1024];
int A[1024][1024][1024][1024];
int B[1024][1024][1024][1024];
int C[1024][1024][1024][1024];
int D[1024][1024][1024][1024];

void contraction7() {

#pragma scop
  for (int a = 0; a < 1024; a++)
    for (int b = 0; b < 1024; b++) 
      for (int c = 0; c < 1024; c++)
	for (int d = 0; d < 1024; d++)
	  for (int e = 0; e < 1024; e++)
	    for (int f = 0; f < 1024; f++)
	      for (int i = 0; i < 1024; i++)
		for (int j = 0; j < 1024; j++)
		  for (int k = 0; k < 1024; k++)
		    for (int l = 0; l < 1024; l++)
		      E[a][b][i][j] += A[a][c][i][k] * B[b][e][f][l] * C[d][f][j][k] * D[c][d][e][l];        
#pragma endscop
}
