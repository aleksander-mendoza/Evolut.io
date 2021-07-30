#include <stdio.h>

__global__ void add(int a, int b, int * c){
    *c = a + b;
}

int main(void){
    char * buffer = 0;
    long length;
    FILE * f = fopen ("test.ptx", "rb");

    if (f)
    {
      fseek (f, 0, SEEK_END);
      length = ftell (f);
      fseek (f, 0, SEEK_SET);
      buffer = malloc (length);
      if (buffer)
      {
        fread (buffer, 1, length, f);
      }
      fclose (f);
    }
    if (buffer){
        printf("%s", buffer);
        int c;
        int * dev_c;
        cudaMalloc((void**)&dev_c, sizeof(int));
        add<<<1,1>>>(2,7,dev_c);
        cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
        printf("Hello world %d\n", c);
    }

    return 0;
}