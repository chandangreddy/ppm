double x[1024];
double y[1024];
double alpha;


void dotproduct() {

#pragma scop
for (int i = 0; i < 1024; i++)
  alpha += x[i] * y[i];
#pragma endscop
}
