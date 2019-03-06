
#include <stdio.h>
#include <iostream>
#include <fstream>

int main(){
printf( "hello world\n" );
//cout<<"helloworld"
char tmp_before[20];
sprintf(tmp_before,"_%04d.jpg",int(1));
printf("%s",tmp_before);
}
